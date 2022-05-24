#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2020
"""
import numpy as np
import os
import sys
import logging
from astropy.io import fits
from craco.cardcap import get_single_packet_dtype, NFPGA, NCHAN, CardcapFile


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-c','--count', help='Number of packets to save (-1 is all)', default=-1, type=int)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    for f in values.files:
        ccap = CardcapFile(f)
        packets = ccap.load_packets(count=values.count)
        fout = f.replace('.fits','.npy')
        print(f'saving {f} to {fout}')
        np.save(fout, packets)

if __name__ == '__main__':
    _main()
