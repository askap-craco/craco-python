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

NANT = 36

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
    def __init__(self, ant_mask=None, frac_onsource=1.0, nback:int=2):
        '''
        :ant_mask: is a len(antmask) npy boolean array that is True if the antenna needs to be in the array
        If None we expect all antennas to be unflagged to a scan

        :frac_onsource: Fraction of unmasked antennas to be onsource before we start anyway.

        '''
        self.last_meta = collections.deque(maxlen=nback)
        self.running = False
        self._start_scan_metadata = None
        self._start_scan_mfile = None
        self.nback = nback
        self._ant_mask = np.ones(NANT, dtype=bool) if ant_mask is None else ant_mask.copy()
        assert len(self._ant_mask) == NANT      

        
        self._frac_onsource = float(frac_onsource)
        assert 0 < self._frac_onsource <= 1,' Frac_onsource has to be in (0, 1.0]'


    def push_data(self, d):
        '''
        Push data into this ScanManager and update internal variables.
    

        :returns: running - whether a scan can run or not.

        '''
        # grows until maxlen size, then stays that size and old data falls off. 
        # append() appends on the right so last_meta.append(x)  and last_meta[-1] == x is True
        self.last_meta.append(d) 
        isfull = len(self.last_meta) == self.last_meta.maxlen
        
        mlist = self.last_meta
        sbid_unchanged = all([s['sbid'] == d['sbid'] for s in mlist])
        scan_unchanged  = all([s['scan_id'] == d['scan_id'] for s in mlist])
        mfile = MetadataFile(list(mlist))
        flags_unchanged = np.all(mfile.anyflag[np.newaxis,:] == mfile.anyflag[-1,])

        antflags = mfile.anyflag[-1] # len=36, True if it's flagged
        antok = ~antflags
        ok_ants = self._ant_mask & antok
        num_ok_ants = sum(ok_ants)
        frac_ok_ants = num_ok_ants / sum(self._ant_mask)
        flags_ok = frac_ok_ants  >= self._frac_onsource
        all_unchanged =  isfull and sbid_unchanged and scan_unchanged and flags_unchanged 
        ok_to_run = flags_ok and all_unchanged

        #print(sum(antok), sum(ok_ants), frac_ok_ants, flags_ok, all_unchanged, ok_to_run)
        log.debug('Got %d/%d good ants. Frac OK=%d flags ok? %s all_unchanged=%s ok to run? %s running?', \
                  num_ok_ants, sum(self._ant_mask), frac_ok_ants, flags_ok, all_unchanged, ok_to_run)
        

        if self.running:
            if ok_to_run:
                pass # continue - everything is fine
            else:
                self._stop_scan(d)
        else: # not yet running
            if ok_to_run:
                self._start_scan(d, mfile)
            else:
                pass # Waiting for things to stabilise

        return self.running

    def _start_scan(self, d, mfile):
        '''
        Start scan with given data.
        Basically save the data and a few tidbits
        '''
        self.running = True
        self._start_scan_metadata = d
        self._start_scan_mfile = mfile
    
    def _stop_scan(self, d):
        self.running = False
        self._start_scan_metadata = None

    def _get_meta(self, key):
        return self._start_scan_metadata[key] if self.running else None


    @property
    def sbid(self):
        '''
        Returns SBID if running
        else none
        '''
        return self._get_meta('sbid')
    
    @property
    def scan_id(self):
        '''
        Returns scan_id if runnign
        else none
        '''
        return self._get_meta('scan_id')
    

        


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
