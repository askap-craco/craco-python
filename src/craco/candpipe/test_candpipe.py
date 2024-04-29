#!/usr/bin/env python
"""
Run tests for candpipe
Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import logging
import pytest
import yaml
from craco.candpipe.candpipe import *

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
    example_cands = pd.read_csv('testdata/candpipe/pulsar/SB61584.n500.candidates.b24.txt.uniq.csv')
    assert check_identical(cands, example_cands) == True, f'Missing candidates'
    assert check_identical(example_cands, cands) == True, f'Extra candidates'

def test_candpipe_runs_anti_alias(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/alias/candidates.b25.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    assert len(pipe.steps) == 5 #check its actually runnign the anti aliasing
    cands = pipe.run()

    # Yuanming writes something that in the end does
    # example_cands = pd.read_csv('testdata/candpipe/alias/SB61585.no_alias_filtering.candidates.b25.txt.uniq.csv')
    example_cands = pd.read_csv('testdata/candpipe/alias/candidates.b25.txt.uniq.csv')
    assert check_identical(cands, example_cands, keyname='ALIAS_name') == True, f'Missing candidates'
    assert check_identical(example_cands, cands, keyname='ALIAS_name') == True, f'Extra candidates'



def check_identical(data1, data2,  keyname='PSR_name'):
    '''
    # # check number of candidates in both files
    # snr = 8
    
    # for ind, candfile in enumerate(candfiles):
    #     candfile1 = candfile.replace('test2', 'test1')
    #     candfile2 = candfile
    #     print()
    #     print('cand1', candfile1)
    #     print('cand2', candfile2)
    
    #     cand1 = read_file(candfile1, snr=snr)
    #     cand2 = read_file(candfile2, snr=snr)
    
    #     print(ind, len(cand1), len(cand2))
    #     print('check if any cand1 not in cand2')
    #     check_identical(cand1, cand2)
    #     print('check if any cand2 not in cand1')
    #     check_identical(cand2, cand1)
    '''

    for i in range(len(data1)):

        lpix = data1.iloc[i]['lpix']
        mpix = data1.iloc[i]['mpix']
        tsamp = data1.iloc[i]['total_sample']
        dm = data1.iloc[i]['dm']
        iblk = data1.iloc[i]['iblk']
        category = data1.iloc[i][keyname]

        if category is None or pd.isna(category):
            print(tsamp, category, 'None or nan')
            ind = sum( (data2['lpix'] == lpix) & \
                        (data2['mpix'] == mpix) & \
                        (data2['total_sample'] == tsamp) & \
                        (data2['dm'] == dm) )
        else:
            ind = sum( (data2['lpix'] == lpix) & \
                        (data2['mpix'] == mpix) & \
                        (data2['total_sample'] == tsamp) & \
                        (data2['dm'] == dm) & \
                        (data2[keyname] == category) )

        cluster_id = data1.iloc[i]['cluster_id']

        if ind == 0:
            print('cand1 is not in cand2, cluster_id', cluster_id, 'total_sample', tsamp)
            return False

    return True 


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
