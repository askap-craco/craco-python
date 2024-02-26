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
from astropy.io import fits
from craco.uvfits_calc11 import UvfitsCalc11
import shutil
from scipy import constants

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_uvfits_works():
    fin = 'testdata/SB053972/b00.uvfits'
    metafile = 'testdata/SB053972/SB53972.json.gz'
    hdulist = fits.open(fin)
    dout='/tmp/test/calc'
    
    shutil.rmtree(dout, ignore_errors=True)
    os.makedirs(dout, exist_ok=True)  
    c = UvfitsCalc11(hdulist, metadata_file=metafile, rundir=dout)
    print('tstart', c.tstart, c.tstart.mjd, c.tstart.iso)
    uvw = c.uvw_array_at_time(c.tstart)
    uvw_m = uvw*constants.c
    print(uvw_m)
    assert np.all(abs(uvw_m) < 7e6) # Should be sensible - earth has a radious of < 7e6 meters
    

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
