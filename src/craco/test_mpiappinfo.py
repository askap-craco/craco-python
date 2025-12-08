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
from craco.mpi_appinfo import *

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
class TestValues:
    def __init__(self):
        self.nfpga_per_rx = 6
        self.nbeams = 36
        self.devices = 'mlx5_0,mlx5_2'
        self.ncards_per_host = None
        self.card = [i+1 for i in range(12)]
        self.block = [4,5,6,7]
        self.fpga = [i for i in range(6)]
        self.dead_cards =  ''
        self.search_beams = [b for b in range(36)]
        self.dump_rankfile = False


class TestPipeInfo:
    def __init__(self):
        self.values = TestValues()        
        self.hosts = [f'skadi-{i:02d}' for i in range(18)]
        self.ranks = []



    def add_rank(self, rank):
        self.ranks.append(rank)
    
              
def test_is_beam_processor_correct():
    assert not CandMgrRankInfo.is_beam_processor 
    assert not ReceiverRankInfo.is_beam_processor
    assert BeamTranRankInfo.is_beam_processor
    assert BeamProcRankInfo.is_beam_processor
    assert PlannerRankInfo.is_beam_processor
    assert BeamCandRankInfo.is_beam_processor
        
def test_populate_ranks():
    pipe_info = TestPipeInfo()
    total_cards = len(pipe_info.values.block)*len(pipe_info.values.card)
    populate_ranks(pipe_info, total_cards)
    
    for r in pipe_info.ranks:
        if r.is_beam_processor:
            assert r.beamid // 18 == r.slot

    

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
