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
import tempfile
from craco.candidate_writer import CandidateWriter
from craco.uvfitsfile_sink import UvFitsFileSink
from craco.mpi_obsinfo import MpiObsInfo
from craco.vissource import VisBlock
from craco.timer import Timer
from craft.parset import Parset
from craco.uvfitsfile_sink import DataPrepper



log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

'''
Number of extra blocks to add before and after a cutout
'''
NEXTRA_BLOCKS = 16


class CutoutBuffer:
    def __init__(self, dtype, ndtype, nslots:int, obs_info:MpiObsInfo, maxncand:int=1):
        #self.buf = np.zeros((nslots, ndtype), dtype=dtype)

        # You have to do this as an array of buffers
        # if you do this as a a single np.zeros the transpose runs extremely slowly
        # for some stupid reason (probably alignment)
        # this was a shocking level of pain for time that I'll never get back in my life

        self.buf = [np.zeros(ndtype, dtype=dtype) for i in range(nslots)]
        self.buf_iblk = np.ones(nslots, dtype=int)*-1
        self.write_idx = -1
        self.nslots = nslots
        self.write_iblk = -1
        self.candidates = []
        self.obs_info = obs_info
        assert maxncand > 0
        self.maxncand = maxncand

        # cache fcm
        self.fcm = None
        if self.obs_info is not None:
            self.fcm = Parset.from_file(obs_info.values.fcm)

            # Precompile. But I don't know how to compile stuff ahead of 
            # time without compromosign performance
            #with tempfile.TemporaryDirectory() as d:
            #    cutout_file_name = os.path.join(d, 'testfile.fits')
             #   cutout_file = UvFitsFileSink(self.obs_info, 
             #                           cutout_file_name)
                #cutout_file.compile(self.buf[0]) # doesnt' work. Typing error. can't fix now



        self.vis_nt = dtype['vis'].shape[2] # find vis_nt so we can count blocks

        # compile fast stuff in uvfits - gross
        # actually, this is too hard. Forget it.
        #baselines = list(self.obs_info.baseline_iter())
        #prepper = DataPrepper(None, baselines, self.vis_nt, 0, 0, 1)
        #prepper.compile(VisBlock(self.buf[0], 0, None, None, None))

     

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

    def iblk2slot(self, iblk:int):
        '''
        Convert iblk numeer to slot idx
        raise ValueError if iblk is not in the ringbuffer for some reason
        '''
        nback = self.current_slot_iblk - iblk
        slot_idx = (self.current_slot_idx - nback) % self.nslots
        if self.buf_iblk[slot_idx] != iblk:
            raise ValueError(f'Iblk {iblk} not found at slot {slot_idx} with current slot iblk={self.current_slot_iblk} and idx={self.current_slot_idx}')
        
        return slot_idx
    
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
        dout = self.buf[self.write_idx]



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
        cout = None
        if len(self.candidates) < self.maxncand:
            cout = CandidateOutput(cand, self)
            self.candidates.append(cout)
        
        return cout

    def flush_all(self):
        '''
        Run this if you want to flush all the ringbuffer ot the candidates, if any
        '''
        while len(self.candidates) > 0:
            self.write_next_block()


class CandidateOutput:
    def __init__(self, cand, cutout_buffer:CutoutBuffer, dumpall=False):
        t = Timer()
        self.cand = cand
        self.cutout_buffer = cutout_buffer
        beamid = self.cutout_buffer.obs_info.beamid
        cand_dir = f'beam{beamid:02d}/candidates/iblk{cand["iblk"]}'
        os.makedirs(cand_dir, exist_ok=True)
        t.tick('mkdir')
        candfile = os.path.join(cand_dir, 'candidate.txt')
        candwriter = CandidateWriter(candfile)
        candwriter.write_cands([cand])
        candwriter.close()
        t.tick('write cand')
        # write another version, jsut to see if it works
        #np.savetxt(os.path.join(cand_dir, 'candidate_spacedelim.txt'), np.array([cand]))

        cutout_file_name = os.path.join(cand_dir, 'candidate.uvfits')
        self.cutout_file_name = cutout_file_name
        search_nt = 256
        vis_nt = self.cutout_buffer.vis_nt
        search_blocks_per_input_block = search_nt // vis_nt
        assert search_blocks_per_input_block >= 1

        if dumpall: # dump the entire buffer. Otherwise, try to dump only the bit of the buffer with the FRB in it
            self.start_iblk = self.cutout_buffer.oldest_slot_iblk + 1       
            self.end_iblk = self.cutout_buffer.current_slot_iblk
        else: # dump the FRB only
            # input blocks are 110ms blocks = vis_nt samples. 
            # Candidate 'iblk' is in units of search blocks = 256 samples
            #cand_iblk = cand['iblk']*search_blocks_per_input_block # looks like cand_iblk might be incorrect
            cand_iblk = cand['total_sample'] // vis_nt
            cand_nsamp = cand['dm'] # number of samples of DM delay
            cand_nblks = (cand_nsamp + vis_nt) // vis_nt # number of of blocks of DM delay DM=0 is 1 block
            total_nblks = 2*NEXTRA_BLOCKS + cand_nblks # Total number of blocks, including NEXTRA before and after
                        
            # This is the final block we save. It might not have appeared yet if the detection latency was pretty small
            # I don't think this is a problem. By the time we get to saving it, it'll be there because we only save 1 block
            # at a time
            self.end_iblk = cand_iblk + NEXTRA_BLOCKS

            # begining iblk. Clamp to the oldest available plus 1, because we don't write the first block when we're created.
            # We can't start before that, right!
            self.start_iblk = max(self.end_iblk - total_nblks, self.cutout_buffer.oldest_slot_iblk + 1)
        
        self.start_slot_idx = self.cutout_buffer.iblk2slot(self.start_iblk)
        self.end_slot_idx = self.cutout_buffer.iblk2slot(self.end_iblk)            
        self.curr_slot_idx = self.start_slot_idx
        self.start_samp = self.start_iblk * vis_nt

        hdr = {'IBLKSTRT': (self.start_iblk, 'Starting iblk'),
               'SLTSTRT': (self.start_slot_idx, 'Slot of starting iblk'),
                'IBLKEND': (self.end_iblk, 'Iblk of final block'),
                'SLTEND': (self.end_slot_idx, 'Slot of final blk'),
                'SMPSTRT': (self.start_samp, 'Total sample of first sample in the block')
               }
        for k in CandidateWriter.out_dtype.names:
            s = ('CN'+k.upper()).replace('_','')[:8]
            hdr[s] = (cand[k], f'Candidate {k}')

        t.tick('mkhdr')

        # why don't we just write out as much data as we have? That seems sensible?
        # OK so the current block being written to is self.cutout_buffer.write_iblk

        format = 'fits' # can be 'raw' or 'fits'
        #format = 'raw'
        use_uvws = True
        self.cutout_file = UvFitsFileSink(cutout_buffer.obs_info, cutout_file_name, extra_header=hdr, format=format, use_uvws=use_uvws, fcm=cutout_buffer.fcm)

        t.tick('mkoutfile')
        log.info('Writing candidate %s in %s format to %s from iblk=%s-%s or slot=%s-%s',
                 cand, format, cand_dir, self.start_iblk, 
                 self.end_iblk, self.start_slot_idx, self.end_slot_idx)
        
                
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
        data = self.cutout_buffer.buf[self.curr_slot_idx]
        iblk = self.cutout_buffer.buf_iblk[self.curr_slot_idx]
        info = self.cutout_buffer.obs_info
        block = VisBlock(data['vis'], iblk, info)

        self.cutout_file.write(block)        
        finished = self.curr_slot_idx == self.end_slot_idx
        log.info('Wrote iblk %s to %s, finished=%s', iblk, self.cutout_file_name, finished)
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
