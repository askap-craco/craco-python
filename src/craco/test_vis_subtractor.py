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
from craco.vis_subtractor import VisSubtractor
import pytest

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

nbl = 435
nchan = 256
nt = 256
shape = [nbl, nchan, nt]
np.random.seed(42)

def test_non_integer():
    with pytest.raises(ValueError):
        vs = VisSubtractor(shape, 18)

    with pytest.raises(ValueError):
        vs = VisSubtractor(shape, 511)


def test_integer():
    vs = VisSubtractor(shape, 16) # OK
    vs = VisSubtractor(shape, 256*16) # OK

def test_equal():
    nt = 4
    s = [2,nt]
    d = np.random.randn(2*nt).reshape(s)
    vs = VisSubtractor(s, 4)
    outd = vs(d)
    expect_outd = d - d.mean(axis=-1, keepdims=True) 
    assert np.allclose(outd, expect_outd)

def test_subblocks():
    nt = 4
    s = [1,nt]
    # make 2 blocks. Check it averages out the blocks
    block_size = 2
    d = np.zeros(s)
    d[:,0:block_size] = 1
    d[:, block_size:] = 2
    vs = VisSubtractor(s, block_size)
    outd = vs(d)
    assert np.all(outd == 0)


def test_superblocks():
    # Hmm, this is boring
    nblocks = 2
    nt = 4
    nblkavg = 3
    nbl = 3
    s = np.array([nblocks, nblkavg, nbl, nt])
    d = np.random.randn(s.prod()).reshape(s)
    d = np.arange(s.prod(), dtype=np.complex64).reshape(s)
    vs = VisSubtractor([nbl, nt], nblkavg*nt)

    # use first block for the moment
    expect_avg = d[0,0,:,:].mean(axis=-1, keepdims=True)
    for iblk in range(nblocks):
        for iblkavg in range(nblkavg):
            din = d[iblk, iblkavg, :, :]
            dout = vs(din.copy())

            if iblkavg == nblkavg - 1: # change to full average
                expect_avg = d[iblk,:,:,:].mean(axis=(0,-1), keepdims=True)

            expect_dout = din - expect_avg
            assert np.allclose(expect_dout, dout),f'iblk {iblk} iblkavg={iblkavg}'

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
