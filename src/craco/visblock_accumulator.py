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

class VisblockAccumulatorStruct:
    def __init__(self, nbl, nf, nt):
        shape = (nbl, nf, nt)
        dt = np.dtype([
            ('vis', np.complex64, (nbl,nf,nt)),
            ('tf_weights', bool, (nf,nt)),
            ('bl_weights', bool, (nbl,))
        ])
        self.pipeline_data_array = np.zeros((1), dtype=dt)
        self.pipeline_data = self.pipeline_data_array[0]
        self.dtype = dt
        self.t = 0
        self.nt = nt
        self.mpi_dtype =  mpi4py.util.dtlib.from_numpy_dtype(dtype=dt)
        self.mpi_msg = [self.pipeline_data_array, 
                        self.pipeline_data_array.size, 
                        self.mpi_dtype]
        self.reset() # set bl_weights to True - otherwise preprocess fails with zerodivisionerror


    def write(self, vis_block):
        vis_data = vis_block.data
        assert len(vis_data.shape) >= 4, f'Invalid vis data shape {vis_data.shape} {vis_data.dtype}'
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert vis_data.dtype == np.complex64, f'I think we can only handle complex data in this function. Vis data type was {vis_data.dtype} {vis_data.shape}'
        vis_out = self.pipeline_data['vis']
        output_nt = vis_out.shape[2]
        
        assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
        assert vis_nc*nrx == vis_out.shape[1], f'Output NC should be {self.pipeline_data.shape[1]} but got {vis_nc*nrx} {vis_data.shape}'
        assert vis_out.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'
        blflags = vis_block.baseline_flags # True = Bad, False=Good
        blweights = ~ blflags # True = Good, False = bad
        # If any baselines in the current block are bad, we make them bad for hte whole block
        self.pipeline_data['bl_weights'] &= blweights
        
        tstart = self.t
        tend = tstart + vis_nt

        # loop through each card
        for irx in range(nrx):
            fstart = irx*vis_nc
            fend = fstart + vis_nc
            chan_weights = abs(vis_data[irx, ...]) != 0
            self.pipeline_data['vis'][:,fstart:fend, tstart:tend] = vis_data[irx, ...]
            self.pipeline_data['tf_weights'][fstart:fend, tstart:tend] = np.all(chan_weights == True, axis=0)

        self.t += vis_nt
        assert self.t <= self.nt, f'Wrote too many blocks without reset {self.t} {self.nt}'

    @property
    def msg_size(self):                
        s = self.pipeline_data.nbytes
        return s

    @property
    def is_full(self):
        return self.t == self.nt
    
    def reset(self):
        self.t = 0
        self.pipeline_data['bl_weights'][:] = True # make them all good again
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