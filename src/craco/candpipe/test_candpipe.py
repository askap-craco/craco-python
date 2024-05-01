#!/usr/bin/env python
"""
Run tests for candpipe
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
import yaml
from craco.candpipe.candpipe import *
from craco.plot_cand import load_cands

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

@pytest.fixture
def config():
    config_file = './testdata/candpipe/config.yaml'
    with open(config_file, 'r') as yaml_file:
        c = yaml.safe_load(yaml_file)

    return c

def test_beam_from_candfile():
    f = '/data/craco/craco/wan348/benchmarking/SB61584.n500.candidates.b24.txt'
    assert beam_from_cand_file(f) == 24

    assert beam_from_cand_file('candidates.b24.txt') == 24

def test_candpipe_runs_noalias(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/pulsar/SB61584.n500.candidates.b24.txt'

    args = parser.parse_args([cand_fname])
    pipe = Pipeline.from_candfile(cand_fname, args, config)
    cands = pipe.run()

    # Yuanming writes something that in the end does
    # assert check_identical(cands, other_cands) == True, f'Candi9dates not identical'

def test_candpipe_runs_anti_alias(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    assert len(pipe.steps) == 5 #check its actually runnign the anti aliasing
    cands = pipe.run()

    # Yuanming writes something that in the end does
    # assert check_identical(cands, other_cands) == True, f'Candi9dates not identical'

def cand_blocker(cands):
    if isinstance(cands, str):
        cands = load_cands(cands)
    
    maxblk = max(cands['iblk'])
    for iblk in range(maxblk+1):
        yield cands[cands['iblk'] == iblk]

def test_convert_np_to_df(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    cands = load_cands(cand_fname)
    df = pipe.convert_np_to_df(cands)


def test_candpipe_block_by_block(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    cands = load_cands(cand_fname)

    #all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    all_clustered_cands = pd.concat(all_clustered_cands)
    

    assert len(all_clustered_cands) >= 1, 'Expected at least 1 candidate '
    assert len(all_clustered_cands) < len(cands), 'Should have been less candidates after pipeline!'
        

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
