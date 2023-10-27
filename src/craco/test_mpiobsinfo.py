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
import glob
from collections import namedtuple
from craco.vissource import open_source
from craco.cardcapfile import CardcapFile
import itertools

from craco.mpipipeline import MpiObsInfo, get_parser

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

TestPipeInfo = namedtuple("TestPipeInfo", "is_beam_processor rx_processor_rank0 beamid values")

def test_obs_info_with_card_disabled():
    '''
    block 5 card 4 was disabled with fch1=0
    '''
    
    pattern = '/data/seren*/big/craco/SB053553/scans/00/20231006062841/ccap*.fits'
    files = sorted(glob.glob(pattern))
    assert len(files) >= 6
    #ccaps = [open_source(f) for f in files]

    all_files = [CardcapFile(f) for f in files]
    enable = [f.card_enabled for f in all_files]
    assert not np.all(enable == False)

    all_hdrs = []

    for k,g in itertools.groupby(all_files, key=lambda f:(f.shelf, f.card)):
        all_hdrs.append([str(f.mainhdr) for f in g])

    values = get_parser().parse_args(['--pol-sum'])
    
    pipe_info = TestPipeInfo(False, 0, 0, values)
    obs_info = MpiObsInfo(all_hdrs, pipe_info)

                       

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
