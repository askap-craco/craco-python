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
from craco.savescan import *
from askap.parset import ParameterSet

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_calc_inttime_tscrunch():
    assert calc_inttime_tscrunch(0) == (16,1)
    assert calc_inttime_tscrunch(1) == (32,1)
    assert calc_inttime_tscrunch(2) == (64,1)
    assert calc_inttime_tscrunch(3) == (64,2)
    assert calc_inttime_tscrunch(4) == (64,4)
    assert calc_inttime_tscrunch(5) == (64,8)
    assert calc_inttime_tscrunch(6) == (64,16)
    assert calc_inttime_tscrunch(7) == (64,32)


@pytest.fixture
def filled_parset():
    '''
    Parset contains this:
    testdata$ grep craco SB071979.params 
    common.enable_craco = true
    craco.archive.location = acacia:as305
    craco.uvfits.int_time_exp = 7
    '''
    p = ParameterSet('testdata/SB071979.params')
    return p

@pytest.fixture
def empty_parset():
    '''
    Parset doesn't contain craco lines
    '''
    p = ParameterSet('testdata/SB071979.params.nocraco')
    return p

def test_filled_parset_with_int_time(filled_parset):
    assert is_parset_specified(filled_parset, 'craco.uvfits.int_time_exp') == True
    int_time_exp = int(get_param_with_default(filled_parset, 'craco.uvfits.int_time_exp', 3))
    assert int_time_exp == 7, 'Should override default of 3'


def test_filled_parset_with_location(filled_parset):
    assert is_parset_specified(filled_parset, 'craco.archive.location') == True
    archive_location  = get_param_with_default(filled_parset, 'craco.archive.location', '')
    assert archive_location == 'acacia:as305', 'Should override default of ""'

def test_empty_parset_with_int_time(empty_parset):
    assert is_parset_specified(empty_parset, 'craco.uvfits.int_time_exp') == False
    int_time_exp = int(get_param_with_default(empty_parset, 'craco.uvfits.int_time_exp', 3))
    assert int_time_exp == 3, 'Should get default of 3'

def test_empty_parset_with_location(empty_parset):
    assert is_parset_specified(empty_parset, 'craco.archive.location') == False
    archive_location  = get_param_with_default(empty_parset, 'craco.archive.location', '')
    assert archive_location == '', 'Should get default of ""'



    



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
