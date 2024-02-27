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
from craft import uvfits
from craco.uvwsource_calc11 import *
import shutil
from scipy import constants

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
fin = 'testdata/SB053972/b00.uvfits'
metafile = 'testdata/SB053972/SB53972.json.gz'

@pytest.fixture
def hdulist():
    return fits.open(fin)
  

def test_uvw_array_at_time_works():    
    dout='/tmp/test/calc'
    f = uvfits.open(fin)
    
    shutil.rmtree(dout, ignore_errors=True)
    os.makedirs(dout, exist_ok=True)
    
    c = UvwSourceCalc11.from_uvfits(f, dout)
    uvw = c.uvw_array_at_time(c.start_time)
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
