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
from pytest import fixture
from craco.injection.packet_injector import *
from craco.cardcap import CardcapFile, get_single_packet_dtype
from numba.typed import List


__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)


nt = 64
nbeam = 36
nant = 30
#nant = 3
nbl = nant*(nant + 1)//2

nfpga = NFPGA
nc = NCHAN*NFPGA
npol = 1
_,_,auto_idxs,cross_idxs = get_indexes(nant)


@fixture
def packets():
    nfpga = 6    
    nt = 32
    npkt = NBEAM*NCHAN
    pktshape = (npkt, nt)
    polsum = True
    debughdr = True
    dtype = get_single_packet_dtype(nbl, debughdr, polsum)
    din_list = [np.zeros(pktshape, dtype=dtype) for i in range(nfpga)]

    for d in din_list:
        d['data'] = (np.random.rand(*d['data'].shape)-0.5)*32000
        d['data'][:,:,:,auto_idxs,:,1] = 0 # autos have 0 imaginary part
        assert not np.all(d['data'] == 0)


    pkts = [pkt for pkt in din_list]
    packet_list = List()
    [packet_list.append(pkt) for pkt in pkts]
    return packet_list


def test_inject_works(packets):
    fids = np.zeros(len(packets))
    valid = np.ones(len(packets), dtype=bool)
    isamp = 0
    phasors = np.ones((nbeam, nbl), dtype=np.complex64)
    chan_delays = np.ones((nbeam, nc), dtype=np.int64) # global channel days. Add DM sweep plus global start time to this.
    tfmap = np.zeros((nc, 10), dtype=np.float32) # can have any number of times. But we choose some smal nunber
    tfmap[:,3] = 1 # amplitude 1 at index=3

    # zero packets
    for pkt in packets:
        pkt['data'] = 0

    inject(packets, valid, isamp, phasors, chan_delays, tfmap)

    # check we've added it in.
    for pkt in packets:
        assert np.all(pkt['data'][:,2,0,:,:,0] == 1) # real
        assert np.all(pkt['data'][:,2,0,:,:,1] == 0) # imag


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
