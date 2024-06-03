#!/usr/bin/env python
"""
Monitors scan metadata and tells you whether we should start or stop a scan

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
    def __init__(self, ant_numbers=None, frac_onsource=1.0, nback:int=2):
        '''
        :ant_numbers: 1 based antenna numbers to include
        :frac_onsource: Fraction of unmasked antennas to be onsource before we start anyway.

        '''

        if ant_numbers is None:
            ant_numbers = np.arange(NANT) + 1

        self.ant_numbers = ant_numbers
        self.last_meta = collections.deque(maxlen=nback)
        self.running = False
        self._start_scan_metadata = None
        self._start_scan_mfile = None
        self.nback = nback
        self._ant_mask = np.zeros(NANT, dtype=bool)
        assert len(self._ant_mask) == NANT        
        self._ant_mask[ant_numbers-1] = True # True is usable      
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
        assert len(mlist) > 0
        mfile = MetadataFile(list(mlist))
        
        flags_changed = mfile.anyflag[:,self._ant_mask] != mfile.anyflag[-1,self._ant_mask]
        flagged_antennas = np.nonzero(mfile.anyflag[-1,:])[0] + 1
        changed_antennas = np.nonzero(flags_changed == True)[0] + 1
        flags_unchanged = np.all(mfile.anyflag[:,self._ant_mask] == mfile.anyflag[-1,self._ant_mask])
        scan_id =  mlist[-1]['scan_id']
        scan_ok = scan_id >= 0

        antflags = mfile.anyflag[-1] # len=36, True if it's flagged
        antok = ~antflags
        ok_ants = self._ant_mask & antok
        num_ok_ants = sum(ok_ants)
        frac_ok_ants = num_ok_ants / sum(self._ant_mask)
        flags_ok = frac_ok_ants  >= self._frac_onsource
        all_unchanged =  isfull and sbid_unchanged and scan_unchanged and flags_unchanged 
        ok_to_run = flags_ok and all_unchanged and scan_ok

        #print(sum(antok), sum(ok_ants), frac_ok_ants, flags_ok, all_unchanged, ok_to_run)
        self.latest_good_metafile = None
        if ok_to_run:
            self.latest_good_metafile = mfile
            
        if self.running:
            if ok_to_run:
                pass
            else:
                self._stop_scan(d)

        else: # not yet running
            if ok_to_run:
                self._start_scan(d, mfile)
            else:
                pass # Waiting for things to stabilise

        log.debug('Got %d/%d good ants. Frac OK=%0.2f flags ok? %s all_unchanged=%s ok to run? %s running? %s changed antennas=%s bad antennas=%s', \
                  num_ok_ants, sum(self._ant_mask), frac_ok_ants, flags_ok, all_unchanged, ok_to_run, self.running, changed_antennas, flagged_antennas)
        
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
    
    @property
    def target_name(self):
        return self._get_meta('target_name')

    @property
    def scan_metadata(self):
        return self._start_scan_mfile
    

        


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
