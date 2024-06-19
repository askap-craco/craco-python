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
from craco import tab_filterbank


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_tab_filterbank_memory_leak():
    #tab_filterbank -uv /CRACO/DATA_01/craco/SB063039/scans/00/20240618010626/b00.uvfits -c /CRACO/DATA_00/craco/SB063039/cal/00/b00.aver.4pol.smooth.npy -t 22h36m58.42529297s -41d05m12.54730225s /tmp/test.b16.fil
    parser = tab_filterbank.get_parser()
    cmdline = ' -uv /CRACO/DATA_01/craco/SB063039/scans/00/20240618010626/b00.uvfits -c /CRACO/DATA_00/craco/SB063039/cal/00/b00.aver.4pol.smooth.npy -t "22h36m58.42529297s -41d05m12.54730225s" /tmp/test.b16.fil'
    values = parser.parse_args(['-uv','/CRACO/DATA_01/craco/SB063039/scans/00/20240618010626/b00.uvfits',
                                '-c', '/CRACO/DATA_00/craco/SB063039/cal/00/b00.aver.4pol.smooth.npy',
                                 '-t', "22h36m58.42529297s -41d05m12.54730225s" ,
                                 '/tmp/test.b16.fil',
                                 '--process_samps','256'])
    tab_filterbank.run(values.files[0], values)
    # should eventually finish without hanging

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
