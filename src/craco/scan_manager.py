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
import collections
from craco.metadatafile import MetadataFile

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

# number of metadatda values to remember in ringbuffer
NBACK = 2
   

class ScanManager:
    '''
    Gets metadata from the metadata service.
    If the last NBACK metadata chunks have equal values of the following, then a scan is initiated.
    If any of them have changed or are invalid, then the scan is quit and stopped.
    - SBID
    - SCANID
    - Antenna flags
    - All flag

    '''
    def __init__(self):
        self.scan_open = False
        self.sbid = None
        self.scanid = None
        self.last_meta = collections.deque(maxlen=NBACK)

    def push_data(self, d):
        # grows until maxlen size, then stays that size and old data falls off. 
        # append() appends on the right so last_meta.append(x)  and last_meta[-1] == x is True
        self.last_meta.append(d) 
        
        mlist = self.last_meta
        sbid_unchanged = all([s['sbid'] == d['sbid'] for s in mlist])
        scan_unchanged  = all([s['scan_id'] == d['scan_id'] for s in mlist])
        mfile = MetadataFile(list(mlist))
        flags_unchanged = np.all(mfile.anyflag[np.newaxis,:] == mfile.anyflag[-1,])
        print(sbid_unchanged, scan_unchanged, flags_unchanged)
        


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
