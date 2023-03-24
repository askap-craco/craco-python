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
import dpkt

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

    fin = values.files[0]

    times = []
    lens = []
    

    for ts, pkt in dpkt.pcap.Reader(open(fin, 'rb')):
        times.append(ts)
        lens.append(len(pkt))


    times = np.array(times)
    lens = np.array(lens)

    tdiff = times[1:] - times[:-1]

    pylab.figure()
    pylab.hist(tdiff, bins=100)
    pylab.show()

    
    
    

if __name__ == '__main__':
    _main()
