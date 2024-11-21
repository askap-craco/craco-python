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
from craco.candidate_writer import CandidateWriter
from craco.uvfitsfile_sink import UvFitsFileSink
from craco.mpi_obsinfo import MpiObsInfo
from craco.vissource import VisBlock

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class CutoutBuffer:
    def __init__(self, dtype, ndtype, nslots:int, obs_info:MpiObsInfo):
        self.buf = np.zeros((nslots, ndtype), dtype=dtype)
        self.buf_iblk = np.ones(nslots, dtype=int)*-1
        self.write_idx = -1
        self.nslots = nslots
        self.write_iblk = -1
        self.candidates = []
        self.obs_info = obs_info

    @property
    def oldest_slot_idx(self):
        '''
        Returns the slot index with the oldest data in it
        If we haven't had nslots data, it will return 0
        '''
        # which means the oldest data is at self.cutout_buffer.write_idx + 1 % nslots
        # if we've had
        if self.write_iblk < self.nslots:
            oldest_slot = 0
        else:
            oldest_slot = (self.write_idx + 1) % self.nslots
        return oldest_slot

    @property
    def oldest_slot_iblk(self):
        '''
        Returns the iblk number of the oldest data.
        May be -1 if no data
        '''
        return self.buf_iblk[self.oldest_slot_idx]

    @property
    def current_slot_idx(self):
        return self.write_idx
    
    @property
    def current_slot_iblk(self):
        return self.buf_iblk[self.current_slot_idx]
    
    def next_write_buffer(self, iblk=None):
        '''
        get next buffer to receive transpose data for
        Increments slot index by 1 mod length
        :iblk: block number - just to check against internal counter
        '''
        self.write_iblk += 1

        if iblk is not None:
            assert iblk == self.write_iblk, 'Current block out of step'
        
        self.write_idx = (self.write_idx + 1) % self.nslots
        self.buf_iblk[self.write_idx] = self.write_iblk
        dout = self.buf[self.write_idx,:]



        return dout
    
    def write_next_block(self):
        '''
        Actualy write a data blocks to disk, if any
        We only dribble data out one block at a time so we don't get blocked in the fwrite
        '''
        # operate on bufer and remove if needed
        cands = self.candidates[:]
        for candout in cands:
            finished = candout.write_next_block()
            if finished:
                self.candidates.remove(candout)



    
    def add_candidate_to_dump(self, cand):
        '''
        Add this candidate to the list of candidates to dump and write to disk
        '''
        cout = CandidateOutput(cand, self)
        self.candidates += cout
        return cout

    def flush_all(self):
        '''
        Run this if you want to flush all the ringbuffer ot the candidates, if any
        '''
        while len(self.candidates) > 0:
            self.do_writing()


class CandidateOutput:
    def __init__(self, cand, cutout_buffer:CutoutBuffer):
        self.cand = cand
        self.cutout_buffer = cutout_buffer
        beamid = self.cutout_buffer.obs_info.beamid
        cand_dir = f'beamd{beamid:02d}/candidates/iblk{cand["iblk"]}'
        os.makedirs(cand_dir, exist_ok=True)
        candfile = os.path.join(cand_dir, 'candidate.txt')
        candwriter = CandidateWriter(candfile)
        candwriter.write_cands([cand])
        candwriter.close()
        # write another version, jsut to see if it works
        np.savetxt(os.path.join(cand_dir, 'candidate_spacedelim.txt'), np.array([cand]))
        cutout_file_name = os.path.join(cand_dir, 'candidate.uvfits')
        self.start_iblk = self.cutout_buffer.oldest_slot_iblk
        self.start_slot_idx = self.cutout_buffer.oldest_slot_idx
        self.end_iblk = self.cutout_buffer.current_slot_iblk
        self.end_slot_idx = self.cutout_buffer.current_slot_idx
        self.curr_slot_idx = self.start_slot_idx
        hdr = {'START_IBLK': self.start_iblk,
               'START_SLTID': self.start_slot_idx,
                'END_IBLK': self.end_iblk,
                'END_SLTID': self.end_slot_idx,              
               }
        for k in CandidateWriter.out_dtype.names:
            s = 'CAND_'+k.upper()
            hdr[s] = cand[k]

        # why don't we just write out as much data as we have? That seems sensible?
        # OK so the current block being written to is self.cutout_buffer.write_iblk

        self.cutout_file = UvFitsFileSink(cutout_buffer.obs_info, cutout_file_name, extra_header=hdr)
        
                
    @property
    def is_finished(self):
        fin = self.cutout_file is None
        return fin

    def write_next_block(self):
        '''
        Write the next block of data to the output
        '''
        if self.cutout_buffer is None: # shoudln't happen, but quit sillently if we've already finished
            return
        data = self.cutout_buffer[self.curr_slot_idx]
        iblk = self.cutout_buffer.buf_iblk[self.curr_slot_idx]
        info = self.cutout_buffer.obs_info
        block = VisBlock(data, iblk, info)

        self.cutout_file.write(block)
        finished = self.curr_slot_idx == self.end_slot_idx
        if finished:
            self.cutout_file.close()
            self.cutout_file = None
        else:
            # increment slot number. 
            self.curr_slot_idx = (self.curr_slot_idx + 1) % self.cutout_buffer.nslots
        

        return finished
    

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
