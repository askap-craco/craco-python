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
from craco.quicksnip import snip
from IPython import embed
from numpy.testing import assert_allclose


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_snip_works():
    infile = 'testdata/SB053972/b00.uvfits'
    inhdu = fits.open(infile)
    outfile = 'snipped.uvfits'
    snip(infile, outfile, 0, 100)
    outhdu = fits.open(outfile)
    assert len(inhdu) == len(outhdu)
    ind = inhdu[0]
    outd = outhdu[0]
    n = len(outd.data)
    for p in ind.parnames:
        if p != 'DATE':
            assert np.all(ind.data[p][:n] == outd.data[p][:n]), f'{p} not equal'

    assert np.all(ind.data[:n]['DATA'] == outd.data[:n]['DATA'])
    for intab, outtab in zip(inhdu[1:], outhdu[1:]):
        assert np.all(intab.data == outtab.data)

def test_snip_works_with_metadata():
    infile = 'testdata/SB053972/b00.uvfits'
    metafile = 'testdata/SB053972/SB53972.json.gz'
    inhdu = fits.open(infile)
    outfile = 'snipped.uvfits'
    snip(infile, outfile, 0, 100,metafile)
    outhdu = fits.open(outfile)
    assert len(inhdu) == len(outhdu)
    ind = inhdu[0]
    outd = outhdu[0]
    n = len(outd.data)
    for p in ind.parnames:
        if p != 'DATE':
            assert_allclose(ind.data[p][:n], outd.data[p][:n],  rtol=1e-6) #,err_msg=f'{p} not equal')

    # only check data - don't check weights
    assert np.all(ind.data[:n]['DATA'][...,:2] == outd.data[:n]['DATA'][...,:2])
    for intab, outtab in zip(inhdu[1:], outhdu[1:]):
        assert np.all(intab.data == outtab.data)
    


    
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
