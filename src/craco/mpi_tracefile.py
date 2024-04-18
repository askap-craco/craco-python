#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
from craco.tracing.tracing import Tracefile
import atexit

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class MpiTracefile:
    '''
    Singletone class that creates a tracefile named by the MPI rank.
    This can be used to add events.
    A little tricky to work out how to close it, but we'll work that out.
    This is a singleton, so you can log to it from anywhere.
    '''
    # How to make a singleton class
    #https://medium.com/@yeaske/singleton-pattern-in-python-a-beginners-guide-75e97ce75554

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.filename = f'rank_{self.rank:03d}.trace'
        self.tracefile  = Tracefile(self.filename, 'array')
        atexit.register(self.close)

    def close(self):
        self.tracefile.close()







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
