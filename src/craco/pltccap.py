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
from craco.cardcapfile import CardcapFile

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def plot(f, values):
    cc = CardcapFile(f)
    fid, d = next(cc.frame_iter())
    print(f, fid, d.shape, d.dtype, d['data'].shape)
    (products, revproducts, auto_products, cross_products) = cc.indexes
    (nc, nt1) = d.shape
    nt2, nbl, npol, _   = d[0,0]['data'].shape
    ds = d['data'].reshape(nc, nt1*nt2, nbl, npol, 2)
    print('ds', ds.shape)
    autos = ds[:,:,auto_products,:,0]
    print(autos.shape)
    for pol in range(npol):
        pylab.plot(autos[...,pol].mean(axis=2).flatten(), label=f'mean pol={pol}')
        pylab.plot(autos[...,pol].std(axis=2).flatten(), label=f'std pol={pol}')
        
    pylab.xlabel('ant x chan')
    pylab.legend()
    pylab.title(f)
    pylab.show()
    
    
    

    

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


    for f in values.files:
        plot(f, values)
    

if __name__ == '__main__':
    _main()
