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
from craco.injection.injection_generator import *


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

def test_injector_makes_sense():
    igen = dm0_pc_regular(nbeam, nbl, nc, delta_t=256, delta_beam=1)
    for i, inj in enumerate(igen):
        print(f'Injection {i} - {inj.end_samp}')



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
