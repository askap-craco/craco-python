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

class MyVisBlock:
    def __init__(self, nbl, nrx, vis_nc, vis_nt):
        self.data = np.zeros((nrx, nbl, vis_nc, vis_nt), dtype=np.complex64)
        self.baseline_flags = np.zeros(nbl, dtype=np.uint8)


def test_visblock_accum():
    nant = 24
    nbl = nant*(nant-1)//2
    nchan = 288
    nt = 256
    nrx = 72
    vis_nt = 32
    vis_nc = nchan // nrx
    vs = VisblockAccumulatorStruct(nbl, nchan, nt)
    block = MyVisBlock(nbl, nrx, vis_nc, vis_nt)
    nblocks = nt // vis_nt

    for iblk in range(nblocks):
        vs.write(block) 

def test_compare_slow_and_fast():
    nant = 24
    nbl = nant*(nant-1)//2
    nchan = 288
    nt = 256
    nrx = 72
    vis_nt = 32
    vis_nc = nchan // nrx
    vs1 = VisblockAccumulatorStruct(nbl, nchan, nt)
    vs2 = VisblockAccumulatorStruct(nbl, nchan, nt)
    block = MyVisBlock(nbl, nrx, vis_nc, vis_nt)
    block.data[:].real = np.random.randn(*block.data.shape)
    block.data[:].imag = np.random.randn(*block.data.shape)
    block.baseline_flags[:] = np.random.randint(0,2, size=nbl)

    write_slow(0,block.data,block.baseline_flags,vs1.pipeline_data)
    write_fast(0,block.data,block.baseline_flags,vs2.pipeline_data)

    assert np.all(vs1.pipeline_data == vs2.pipeline_data)


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
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
