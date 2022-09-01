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


    for fname in values.files:
        f = CardcapFile(fname)
        pkts = f.load_packets()
        fig, ax = pylab.subplots(2,2, sharex=True)
        fig.suptitle(fname)
        ax = ax.flatten()
        ax[0].plot(pkts['frame_id'])
        ax[1].plot(pkts['bat'])
        ax[2].plot(pkts['beam_number'])
        ax[3].plot(pkts['channel_number'])

        ax[0].set_ylabel('frame_id')
        ax[1].set_ylabel('bat')
        ax[2].set_ylabel('beam')
        ax[3].set_ylabel('channel')
        pylab.show()

if __name__ == '__main__':
    _main()
