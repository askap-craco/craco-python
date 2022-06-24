#!/usr/bin/env python
"""
Class to merge cardcap files

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco.cardcap import CardcapFile, NCHAN

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def none_next(i):
    try:
        b = next(i)
    except StopIteration:
        b = None

    return b

def frame_id_iter(i, fid0, fidoff):
    '''
    Generator that yields blocks of data
    It will yield a block with the given frame id startign at fid0 and incrementing
    by fidoff
    if no block has the given frame ID, it will yield None
    '''

    frame_id = fid0
    fidoff = np.uint64(fidoff)
    assert isinstance(fid0, np.uint64)
    currblock = None
    while True:
        if currblock is None or currblock['frame_id'][0] < frame_id:
            currblock = next(i)

        curr_bat = currblock['bat'][0]
        curr_frameid = currblock['frame_id'][0]

        if curr_frameid == frame_id:
            b = currblock
            log.debug(f'HIT frame_id={frame_id} hit {curr_bat}')
        else:
            log.debug(f'MISS frame_id={frame_id} {curr_frameid} {curr_bat}')
            b = None

        frame_id += fidoff
        yield curr_frameid, b

class CcapMerger:
    def __init__(self, fnames):
        self.fnames = fnames
        self.ccap = [CardcapFile(f) for f in self.fnames]
        nfpga = len(self.ccap[0].fpgas)
        nfiles = len(self.ccap)
        all_freqs = np.zeros((nfiles, nfpga, NCHAN))
        frame_ids = [c.frame_id0 for c in self.ccap]
        bats = [c.bat0 for c in self.ccap]
        log.debug('Frame IDs', frame_ids)
        log.debug('bats', bats)
        

        for ic, c in enumerate(self.ccap):
            assert len(c.fpgas) == nfpga, 'Differing numbers of fpgas in the files'
            all_freqs[ic, :, :] = c.frequencies
            
        fidxs = np.argsort(all_freqs.flat).reshape(nfiles, nfpga, NCHAN)
        self.fidxs = fidxs
        self.frame_id0 = min(frame_ids)
        self.tscrunch = self.ccap[0].mainhdr['TSCRUNCH']
        self.nbeam = self.ccap[0].nbeam
        self.nbl = self.ccap[0].mainhdr['NBL']
        self.__npol = self.ccap[0].npol
        self.all_freqs = all_freqs
        self.nint = 1 # TODO: Fix

    @property
    def fcent(self):
        '''
        Returns center frequency - units of MHz
        '''
        return self.all_freqs.mean()

    @property
    def foff(self):
        '''
        Returns channel interval - MHz
        '''
        fsort = np.array(sorted(self.all_freqs.flat))
        fdiff = fsort[1:] - fsort[:-1]
        foff = fdiff.mean()
        return foff

    @property
    def nchan(self):
        return self.all_freqs.size

    @property
    def npol(self):
        return self.__npol

    @property
    def mjd0(self):
        t = self.ccap[0].time_of_frame_id(self.frame_id0)
        log.debug(f'MJD0 for {self.ccap[0].fname} is {t}')
        return t

    @property
    def nant(self):
        n = self.ccap[0].mainhdr['NANT']
        return n

    @property
    def inttime(self):
        t = self.ccap[0].mainhdr['TSAMP']*self.ccap[0].mainhdr['TSCRUNCH']
        return t
    
    def fid_to_mjd(self, fid):
        return self.ccap[0].time_of_frame_id(fid)

    def block_iter(self):
        '''
        Returns an iterator that returns blocks of data
        Blocks have shape (nchan,nbeam,ntime,nbl,npol,2), dtype=np.int16 and are masked arrays
        Mask is true (invalid) if frameID missing from file, or file has terminated
        '''
        packets_per_block = 36*4 # TODO: work out how to work this out
        fidoff = 2048

        iters = [frame_id_iter(c.packet_iter(packets_per_block), self.frame_id0, fidoff) for c in self.ccap]
        
        while True:
            packets = []
            finished_array = []
            for i in iters:
                try:
                    packets.append(next(i))
                    finished_array.append(False)
                except StopIteration:
                    finished_array.append(True)
                    
            flagged_array = [p[1] is None for p in packets]
            fids = [p[0] for p in packets]
            finished = all(finished_array)
            log.debug('Finished %s %s - flagged %s FIDS=%s', finished, finished_array, flagged_array, fids)
            
            if finished:
                break

            shape = (self.nchan, self.nbeam, self.nint, self.nbl, self.npol, 2)
            dout = np.zeros(shape, dtype=np.int16)
            mask = np.zeros(shape, dtype=np.bool)
            for ip, (fid, p) in enumerate(packets):
                assert self.fidxs.shape[1] == 1, 'Can only handle single FPGA files'
                freqidx = self.fidxs[ip,0,:]
                if p is None:
                    mask[freqidx, ...] = True
                    # data is already 0, but now it's masked anyway
                else:
                    # mask is already false = valid data
                    blk1 = p[:32*4]
                    blk1.shape = (4,32)
                    blk2 = p[32*4:]
                    blk2.shape = (4,4)
                    dout[freqidx, :32, ...] = blk1['data']
                    dout[freqidx, 32:, ...] = blk2['data']

            dout = np.ma.masked_array(dout,mask)
                
            yield (fid, dout)
            
            

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

    merger = CcapMerger(values.files)
    for fid, blk in merger.block_iter():
        print(blk.shape)
    
    

if __name__ == '__main__':
    _main()
