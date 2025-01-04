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
from craco.visblock_accumulator import *

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

vis_fscrunch = 6
vis_tscrunch = 1

class MyVisBlock:
    def __init__(self, nbl, nrx, vis_nc, vis_nt):
        self.data = np.zeros((nrx, nbl, vis_nc, vis_nt), dtype=np.complex64)
        self.baseline_flags = np.zeros(nbl, dtype=np.uint8)
        self.ics = np.zeros((nrx, vis_nt*vis_tscrunch, vis_nc*vis_fscrunch), dtype=np.float32) # un-fscrunched version


def test_visblock_accum():
    nant = 24
    nbl = nant*(nant-1)//2
    nchan = 288
    nt = 256
    nrx = 72
    vis_nt = 32
    vis_nc = nchan // nrx
    vs = VisblockAccumulatorStruct(nbl, nchan, nt, vis_tscrunch, vis_fscrunch)
    block = MyVisBlock(nbl, nrx, vis_nc, vis_nt)
    nblocks = nt // vis_nt

    for iblk in range(nblocks):
        vs.write(block)

    vs.finalise_weights()

def test_scrunch_ics():
    nrx = 72
    vis_nc = 24
    vis_nt = 32
    nt = 256
    nc = vis_nc * nrx
    ics_data = np.arange(nrx*vis_nc*nt).reshape(nrx,nt,vis_nc)
    expected = ics_data.transpose(0,2,1).reshape(nc,nt)
    expected = expected.reshape(nc//vis_fscrunch, vis_fscrunch, nt).sum(axis=1)
    scrunched_ics = np.zeros((nc // vis_fscrunch, nt), dtype=np.float32)
    for i in range(nt // vis_nt):
        tstart = i * vis_nt
        scrunch_ics(tstart, ics_data[:,i*vis_nt:(i+1)*vis_nt,:],scrunched_ics, vis_tscrunch, vis_fscrunch)

    np.testing.assert_allclose(scrunched_ics, expected)


def test_compare_slow_and_fast():
    nant = 24
    nbl = nant*(nant-1)//2
    nchan = 288
    nt = 256
    nrx = 72
    vis_nt = 32
    vis_nc = nchan // nrx
    vs1 = VisblockAccumulatorStruct(nbl, nchan, nt, vis_tscrunch, vis_fscrunch)
    vs2 = VisblockAccumulatorStruct(nbl, nchan, nt, vis_tscrunch, vis_fscrunch)
    block = MyVisBlock(nbl, nrx, vis_nc, vis_nt)
    block.data[:].real = np.rint(np.random.randn(*block.data.shape))
    block.data[:].imag = np.rint(np.random.randn(*block.data.shape))
    block.baseline_flags[:] = np.random.randint(0,2, size=nbl)

    write_slow(0,block.data,block.baseline_flags,vs1.pipeline_data)
    write_fast(0,block.data,block.baseline_flags,vs2.pipeline_data)

    np.testing.assert_array_equal(vs1.pipeline_data['vis'], vs2.pipeline_data['vis'])
    np.testing.assert_array_equal(vs1.pipeline_data['tf_weights'], vs2.pipeline_data['tf_weights'])
    np.testing.assert_array_equal(vs1.pipeline_data['bl_weights'], vs2.pipeline_data['bl_weights'])

def test_bool_and_uint8_views_equivalent():
    '''
    This is important because in search_pipeline_sink I make a boolean view of the
    uint8 to send to vivek's fastbaseline script
    '''
    n = 256
    dtype = np.uint8
    u = np.random.randint(0,2, size=n).astype(dtype)
    assert np.all((u == 0) | (u == 1))
    assert u.size == n
    b = u.view(dtype=bool)
    assert b.shape == u.shape
    assert np.all(u[b] == 1), 'All uint8s which cast to True should be 1'
    assert np.all(u[~b] == 0), 'All uint8s which cast to False should b 0'


    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    test_compare_slow_and_fast()

    

if __name__ == '__main__':
    _main()
