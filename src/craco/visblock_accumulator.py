#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.timer import Timer
import mpi4py.util.dtlib
from numba import njit
from mpi4py import MPI
from craco.ics_preprocess import get_ics_masks



log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class VisblockAccumulatorMasked:
    def __init__(self, nbl, nf, nt):
        shape = (nbl, nf, nt)
        self.pipeline_data = np.ma.masked_array(np.zeros(shape, dtype=np.complex64), mask=np.zeros(shape, dtype=bool))
        self.t = 0
        self.nt = nt

    def write(self, vis_block):
        vis_data = vis_block.data
        assert len(vis_data.shape) >= 4, f'Invalid vis data shape {vis_data.shape} {vis_data.dtype}'
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert vis_data.dtype == np.complex64, f'I think we can only handle complex data in this function. Vis data type was {vis_data.dtype} {vis_data.shape}'
        output_nt = self.pipeline_data.shape[2]
        
        assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
        assert vis_nc*nrx == self.pipeline_data.shape[1], f'Output NC should be {self.pipeline_data.shape[1]} but got {vis_nc*nrx} {vis_data.shape}'
        assert self.pipeline_data.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'
        blflags = vis_block.baseline_flags[:,np.newaxis,np.newaxis]
        tstart = self.t
        tend = tstart + vis_nt

        # loop through each card
        for irx in range(nrx):
            fstart = irx*vis_nc
            fend = fstart + vis_nc
            chanmask = abs(vis_data[irx, ...]) == 0
            self.pipeline_data[:,fstart:fend, tstart:tend] = vis_data[irx, ...]
            self.pipeline_data.mask[:,fstart:fend, tstart:tend] = chanmask | blflags

        self.t += vis_nt

    @property
    def msg_size(self):
        # 1 byte for mask plus a bit for extra
        s = self.pipeline_data.size * (self.pipeline_data.itemsize + 1) + 1024 
        return s

    @property
    def is_full(self):
        return self.t == self.nt
    
    def reset(self):
        self.t = 0

    def close(self):
        pass


def write_slow(tstart:int, vis_data, blflags, pipeline_data):
    '''
    vis_data is (nrx,nbl, vis_nc, vis_nt)
    bl_flags is lenght (nbl) bool
    pipeline_data is see VisAccumulatorStruct
    '''
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]

    blweights = ~ blflags # True = Good, False = bad
    # If any baselines in the current block are bad, we make them bad for hte whole block
    pipeline_data['bl_weights'] &= blweights    
    tend = tstart + vis_nt

    # loop through each card
    for irx in range(nrx):
        fstart = irx*vis_nc
        fend = fstart + vis_nc
        chan_weights = abs(vis_data[irx, ...]) != 0
        pipeline_data['vis'][:,fstart:fend, tstart:tend] = vis_data[irx, ...]
        pipeline_data['tf_weights'][fstart:fend, tstart:tend] = np.all(chan_weights == True, axis=0)

@njit(cache=True)
def write_fast(tstart:int, vis_data, blflags, pipeline_data):
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
    # If any baselines in the current block are bad, we make them bad for hte whole block
    
    # update 
    # pipeline_data['bl_weights'] &= ~ blflags I think this allocates memory, which sux
    blweights = pipeline_data['bl_weights']
    for ibl in range(nbl):
        blweights[ibl] &= not blflags[ibl]

    tend = tstart + vis_nt

    # loop through each card
    for irx in range(nrx):
        fstart = irx*vis_nc
        fend = fstart + vis_nc
        for ichan, chan in enumerate(range(fstart, fend)):
            for itime, time in enumerate(range(tstart, tend)):
                weight = np.uint8(1) # weight is True if any values are nonzero
                for ibl in range(nbl):
                    d = vis_data[irx, ibl, ichan, itime]
                    # copy data in        

                    pipeline_data['vis'][ibl, chan, time] = d
                    weight &= ((d.real != 0) | (d.imag != 0))
                
                pipeline_data['tf_weights'][chan,time] = weight

        
        #pipeline_data['vis'][:,fstart:fend, tstart:tend] = vis_data[irx, ...]

        # weight if true if all baselines for that cell are nonzero
        #chan_weights = abs(vis_data[irx, ...]) != 0
        #pipeline_data['tf_weights'][fstart:fend, tstart:tend] = np.all(chan_weights == True, axis=0)

@njit(cache=True)
def write_ics(tstart:int, ics_data, all_ics):
    '''
    Writes ICS and transposes
    ICS is [nrx, nt, nc]
    VIS is [nrx, nbl, nc, nt]
    '''
    nrx, ics_nt, ics_nc = ics_data.shape
    for irx in range(nrx):
        for ichan in range(ics_nc):
            ochan = ichan + irx*ics_nc            
            for itime in range(ics_nt):                
                otime = tstart + itime
                all_ics[ochan, otime] = ics_data[irx, itime, ichan]

@njit(cache=True)
def scrunch_ics(tstart:int, ics_data, scrunched_ics, vis_tscrunch:int, vis_fscrunch:int):
    nrx, ics_nt, ics_nc = ics_data.shape
    for irx in range(nrx):
        for ichan in range(ics_nc):
            chan = ichan + irx*ics_nc
            ochan = chan // vis_fscrunch
            for itime in range(ics_nt):                
                otime = (itime + tstart) // vis_tscrunch
                din = ics_data[irx, itime, ichan]
                scrunched_ics[ochan, otime] += din


def allocate_shared_buffer(dt, nblocks, comm):
    disp_unit = dt.itemsize
    win = MPI.Win.Allocate_shared(nblocks * disp_unit if comm.rank == 0 else 0, disp_unit, comm = comm)
    buf, itemsize = win.Shared_query(0)
    assert itemsize == dt.itemsize            
    d = np.ndarray(buffer=buf, dtype=dt, shape=(nblocks,))
    return d


class VisblockAccumulatorStruct:
    def __init__(self, nbl:int, nf:int, nt:int, vis_tscrunch:int=1, vis_fscrunch:int=1, comm=None, nblocks=1):
        '''
        If comm is not None, alocates a shared memory buffer
        '''
        self.vis_tscrunch = vis_tscrunch
        self.vis_fscrunch = vis_fscrunch
        dt = np.dtype([
            ('vis', np.complex64, (nbl,nf,nt)),
            ('tf_weights', np.uint8, (nf,nt)),
            ('bl_weights', np.uint8, (nbl,))
        ])
        if comm is None:
            d = np.zeros((nblocks), dtype=dt)
        else:
            d = allocate_shared_buffer(dt, nblocks, comm)

        self.pipeline_data_array = d
        self.pipeline_data = self.pipeline_data_array[0]

        self.scrunched_ics = np.zeros((nf,nt), dtype=np.float32) # scrunched version of ICS - used for DM0 flagging
        self.all_ics = np.zeros((nf*vis_fscrunch, nt*vis_tscrunch), dtype=np.float32)
        self.ics_weights = np.zeros((nf,nt), dtype=bool)

        self.dtype = dt
        self.t = 0
        self.nt = nt
        self.mpi_dtype =  mpi4py.util.dtlib.from_numpy_dtype(dtype=dt)
        self.mpi_msg = [self.pipeline_data_array, 
                        self.pipeline_data_array.size, 
                        self.mpi_dtype]
        self.nblocks = nblocks
        self.reset() # set bl_weights to True - otherwise preprocess fails with zerodivisionerror

    def compile(self, vis_block):
        '''
        Run write once and reset
        '''
        assert self.t == 0
        self.write(vis_block)
        self.reset()


    def write(self, vis_block, iblk=0):
        t = Timer()

        vis_data = vis_block.data
        assert len(vis_data.shape) >= 4, f'Invalid vis data shape {vis_data.shape} {vis_data.dtype}'
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert vis_data.dtype == np.complex64, f'I think we can only handle complex data in this function. Vis data type was {vis_data.dtype} {vis_data.shape}'
        vis_out = self.pipeline_data['vis']
        output_nt = vis_out.shape[2]
        
        assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
        assert vis_nc*nrx == vis_out.shape[1], f'Output NC should be first of {self.pipeline_data.shape} but got {vis_nc*nrx} {vis_data.shape}'
        assert vis_out.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'
        #assert vis_nt == self.nt // self.vis_tscrunch, f'Vis NT should be {self.nt//self.vis_tscrunch} but got {vis_nt}. nt= {self.nt} tscrunch={self.vis_tscrunch}'

        tstart = self.t
        vis_data = vis_block.data
        blflags = vis_block.baseline_flags # True = Bad, False=Good
        #write_slow(tstart, vis_data, blflags, self.pipeline_data)
        assert 0<= iblk < self.nblocks
        t.tick('init')
        write_fast(tstart, vis_data, blflags, self.pipeline_data_array[iblk])
        t.tick('write vis')
        write_ics(tstart, vis_block.ics, self.all_ics)
        t.tick('write ics')
        scrunch_ics(tstart, vis_block.ics, self.scrunched_ics, self.vis_tscrunch, self.vis_fscrunch)
        t.tick('scrunch ics')


        self.t += vis_nt
        assert self.t <= self.nt, f'Wrote too many blocks without reset {self.t} {self.nt}'


    @property
    def msg_size(self):                
        s = self.pipeline_data.nbytes
        return s

    @property
    def is_full(self):
        return self.t == self.nt
    
    def finalise_weights(self, iblk):
        get_ics_masks(self.scrunched_ics, self.ics_weights.view(dtype=bool))
        self.pipeline_data_array[iblk]['tf_weights'] *= self.ics_weights

    
    def reset(self, iblk=0):
        self.t = 0
        self.pipeline_data_array[iblk]['bl_weights'][:] = 1 # make them all good again
        self.pipeline_data_array[iblk]['tf_weights'][:] = 1
        self.scrunched_ics[:] = 0
        # Don't need to reset 'vis' and 'tf_weights' as they will be overidden

    def close(self):
        pass


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
