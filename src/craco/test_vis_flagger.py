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
from craco.vis_flagger import VisFlagger
import pytest
from craft import sigproc

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

nbl = 5*6//2
nchan = 32
nt = 64
shape = [nbl, nchan, nt]
np.random.seed(42)
hdr = {'nbits':1, 'nchans':nchan, 'nifs':1, 'tstart':0, 'tsamp':0.11, 'fch1':700, 'foff':1}
test_mask_fil_writer = sigproc.SigprocFile("test_masks.fil", 'wb', hdr)

def test_runs():
    shape = (nbl, nchan, nt)
    casshape = (nchan*6, nt)

    vis = np.random.randn(*shape) + 1j*np.random.randn(*shape)
    vis = np.ma.masked_array(vis, vis==0)
    ics = np.random.randn(*casshape)
    cas = np.random.randn(*casshape)

    flagger = VisFlagger(4,4,5, 5)
    visout = flagger(vis, cas, ics, test_mask_fil_writer)

    flagger = VisFlagger(4,0,5, 5)
    visout = flagger(vis,cas, ics, test_mask_fil_writer)

    test_mask_fil_writer.fin.close()
    
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

    test_runs()
    

if __name__ == '__main__':
    _main()
