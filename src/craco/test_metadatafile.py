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
from craco.metadatafile import *
from astropy.coordinates import SkyCoord

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_dummy():
    c=SkyCoord(ra=30,dec=-45,unit='deg',equinox='J2000',frame='icrs')
    name = 'TESTING'
    d = MetadataDummy(name, c)

    sources = d.sources(0)
    src = sources[name]
    assert src['name'] == name
    assert src['skycoord'] == c
    assert src['ra'] == c.ra.deg
    assert src['dec'] == c.dec.deg

    mjd = 235135
    assert d.source_index_at_time(mjd) == 0
    assert d.source_at_time(0,mjd) is not None
    assert np.all(d.uvw_at_time(mjd) == 0)
    assert np.all(d.flags_at_time(mjd) == False)

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
