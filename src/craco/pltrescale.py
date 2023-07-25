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

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def plot(f, values):
    d = np.load(f, values)
    print(f)
    for k in d.keys():
        print(k, d[k].shape, d[k].dtype)

    beam = values.beam
    print(beam)
    fig,ax = pylab.subplots(3,1)
    dcm = d['mean'][...,0] + 1j*d['mean'][...,1]
    ax[0].plot(abs(dcm)[beam,:,:])
    ax[0].set_ylabel('abs(mean)')

    ax[1].plot(np.angle(dcm)[beam,:,:])
    ax[1].set_ylabel('angle(mean')
    
    ax[2].plot(d['stdev'][beam,:,:,0])
    ax[2].set_ylabel('stdev of real')
    ax[2].set_xlabel('channel')
    fig.suptitle(f'{f} beam={beam}')
    pylab.show()

        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-b','--beam', type=int, help='Beam to plot', default=0)
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
