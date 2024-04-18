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
from craco.metadatafile import MetadataFile
from craco.scan_manager import ScanManager

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

metafile = 'testdata/SB053972/SB53972.json.gz'

def test_start_stop_with_frac_ant():
    mfile = MetadataFile(metafile)
    mgr = ScanManager(frac_onsource=0.5)
    for i, d in enumerate(mfile.data):
        mgr.push_data(d)
        print(i, mgr.running)

# TODO: Parse the file directly to get thse numbers
        if 7 <= i <= 62:
            assert mgr.running
            assert mgr.sbid == 53972
            assert mgr.scan_id == 0
        else: # need to zoom into the thign to see the falags only work out at i=7
            assert mgr.running == False
            assert mgr.sbid is None
            assert mgr.scan_id is None

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
