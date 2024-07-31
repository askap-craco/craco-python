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
