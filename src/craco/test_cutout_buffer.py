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
import pytest
from craco.cutout_buffer import *
import craco.card_averager
from craco.candidate_writer import CandidateWriter

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class DummyInfo:
    def __init__(self):
        self.beamid = 3
        

def make_cb(nslots=128):
    nbeam = 36
    nant = 12
    nc = 24
    nt = 32
    vis_fscrunch = 6
    vis_tscrunch = 1
    npol = 1
    real_dtype=np.float32
    cplx_dtype = np.float32
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, real_dtype, cplx_dtype)
    nrx = 72
    buf = np.zeros(nrx, dtype=dt)
    dtall = buf.dtype
    nslots = nslots
    info = DummyInfo()

    c = CutoutBuffer(dtall, nrx, nslots, info)
    return c


def test_check_buffer_dtypes():
    nbeam = 36
    nant = 12
    nc = 24
    nt = 32
    vis_fscrunch = 6
    vis_tscrunch = 1
    npol = 1
    real_dtype=np.float32
    cplx_dtype = np.float32
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, real_dtype, cplx_dtype)
    nrx = 72
    # transposer makes this dtype
    transpose_buffer = np.zeros(nrx, dtype=dt)
    cutout_dtype = transpose_buffer.dtype
    nslots = 10
    info = DummyInfo()
    c = CutoutBuffer(cutout_dtype, nrx, nslots, info)
    write_buffer = c.next_write_buffer()
    # are the dtypes sensible?
    write_buffer_dtype = write_buffer.dtype

    print(transpose_buffer.shape, transpose_buffer.dtype)
    print(c.buf.shape, c.buf.dtype)
    print(write_buffer.shape, write_buffer.dtype)

    assert transpose_buffer.dtype == write_buffer.dtype
    assert transpose_buffer.shape == write_buffer.shape



def test_buffer_slotidx():
    cb = make_cb(nslots=3)

    cand = np.zeros(1, dtype=CandidateWriter.out_dtype)[0]
    cand['snr'] = 0
    cand['iblk'] = 2
    cand['ibeam'] = 3
    cand['mjd'] = 123556.2234
    cand['total_sample'] = 123


    assert cb.current_slot_idx == -1
    assert cb.current_slot_iblk == -1
    assert cb.oldest_slot_idx == 0
    assert cb.oldest_slot_iblk == -1
    assert len(cb.candidates) == 0


    iblk = 0
    buf = cb.next_write_buffer()
    buf['vis'][:] = 0
    cb.write_next_block() # should not fail

    assert cb.current_slot_idx == 0
    assert cb.current_slot_iblk == iblk
    assert cb.oldest_slot_idx == 0
    assert cb.oldest_slot_iblk == 0


    iblk = 1
    buf = cb.next_write_buffer()
    buf['vis'][:] = 1
    cb.write_next_block() # should not fail


    assert cb.current_slot_idx == 1
    assert cb.current_slot_iblk == iblk
    assert cb.oldest_slot_idx == 0
    assert cb.oldest_slot_iblk == 0

    iblk = 2
    buf = cb.next_write_buffer()
    buf['vis'][:] = 2
    cb.write_next_block() # should not fail


    assert cb.current_slot_idx == 2
    assert cb.current_slot_iblk == iblk
    assert cb.oldest_slot_idx == 0
    assert cb.oldest_slot_iblk == 0

    iblk = 3
    buf = cb.next_write_buffer()
    buf['vis'][:] = 3
    cb.write_next_block() # should not fail

    assert cb.current_slot_idx == 0
    assert cb.current_slot_iblk == 3
    assert cb.oldest_slot_idx == 1
    assert cb.oldest_slot_iblk == 1

    # add a candidate now
    # we should get a dump of the iblk = 1, 2, and 3 in the file
    # it should take 3 write_next_block() to finish


    # OMG - setting this up to unit test isa massive headache
    # need to setup an mpiobsinfo with realistic data, which is a complete
    # and utter headach. Bluck.
    # giving up for now
    return

    cout = cb.add_candidate_to_dump(cand)
    assert len(cb.candidates) == 1
    
    # should write iblk = 1
    cb.write_next_block()
    assert len(cb.candidates) == 1

    # should write iblk = 2
    cb.write_next_block()
    assert len(cb.candidates) == 1

    # should write iblk = 3
    # and then write the data
    # and close and flush the buffer
    # and be finished
    cb.write_next_block()
    assert len(cb.candidates) == 0
    assert cout.is_finished
    assert cout.cutout_file is None








    
    


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
