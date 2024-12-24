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

class MpiRingBufferManager:
    '''
    Pretends to manage a ringbuffer via send/recv
    in an asynchronous way
    '''
    def __init__(self, comm, nblocks:int, tx_rank:int=0, rx_rank:int=1):
        self.nblocks = nblocks
        self.comm = comm
        self.widx = -1 # write index
        self.tx_requests = [None for i in range(nblocks)]
        self.rx_requests = [None for i in range(nblocks)]

        self.release_requests = [None for i in range(nblocks)]
        self.ridx = -1 # read index

        self.tx_rank = tx_rank
        self.rx_rank = rx_rank

    def _update_requests(self, requests):
        for ir, r in enumerate(requests):
            if r is not None:
                complete, result = r.test()
                if complete:
                    requests[ir] = None

    def _check_writable(self):
        self._update_requests(self.tx_requests)
        self._update_requests(self.rx_requests)
        assert self.tx_requests[self.widx] is None, f'Outstanding tx request {self.widx}'
        assert self.rx_requests[self.widx] is None, f'Outstanding rx request {self.widx}'

    def open_write(self):
        '''
        Called by writer to open a block for writing
        :returns: index of write slot
        '''
        self.widx = (self.widx + 1) % self.nblocks
        self._check_writable()
        return self.widx

    def close_write(self):
        '''
        Called by writer to close block for writing. Will send info to receiver
        '''
        assert self.widx >= 0
        self._check_writable()        
        self.tx_requests[self.widx] = self.comm.isend(self.widx, dest=self.rx_rank)
        self.rx_requests[self.widx] = self.comm.irecv(source=self.rx_rank)        

    def open_read(self):
        '''
        Called by receiver to read a block
        :returns: read index
        '''
        assert self.ridx == -1, f'Havnt released block {self.ridx}'
        ridx = self.comm.recv(source=self.tx_rank)
        self.ridx = ridx
        return ridx
    
    def close_read(self):
        '''
        Called by receiver to release most recently read block
        '''
        self._update_requests(self.release_requests)
        assert self.release_requests[self.ridx] is None, f'Outstanding release request {self.ridx}'
        self.release_requests[self.ridx] = self.comm.isend(self.ridx, dest=self.tx_rank)
        self.ridx = -1


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
