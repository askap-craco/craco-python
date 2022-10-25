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
from craco.cardcap import CardcapFile

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def getfirst(pkts, fieldname):
    try:
        v = pkts[fieldname][0]
    except IndexError:
        v = '---'

    return v
    

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

    fig, ax = pylab.subplots(2,2, sharex=True)


    for fname in values.files:
        f = CardcapFile(fname)
        pkts = f.load_packets()
        #fig.suptitle(fname)
        ax = ax.flatten()
        print(fname, getfirst(pkts, 'frame_id'), getfirst(pkts, 'bat'), getfirst(pkts, 'beam_number'), getfirst(pkts, 'channel_number'))
        ax[0].plot(pkts['frame_id'], label=fname)
        ax[1].plot(pkts['bat'])
        ax[2].plot(pkts['beam_number'])
        ax[3].plot(pkts['channel_number'])
        ax[0].legend()

        ax[0].set_ylabel('frame_id')
        ax[1].set_ylabel('bat')
        ax[2].set_ylabel('beam')
        ax[3].set_ylabel('channel')
    pylab.show()

if __name__ == '__main__':
    _main()
