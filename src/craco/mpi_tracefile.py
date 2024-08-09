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
    Singleton class that creates a tracefile named by the MPI rank.
    This can be used to add events.
    A little tricky to work out how to close it, but we'll work that out.
    This is a singleton, so you can log to it from anywhere.
    '''
    # How to make a singleton class
    #https://medium.com/@yeaske/singleton-pattern-in-python-a-beginners-guide-75e97ce75554

    _instance = None

    def __init__(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.filename = f'rank_{self.rank:03d}_trace.json'
        self.__tracefile  = Tracefile(self.filename, 'array')
        log.info('Opened tracefile %s', self.filename)
        atexit.register(self.close)

    @staticmethod
    def instance():
        if MpiTracefile._instance is None:
            MpiTracefile._instance = MpiTracefile()

        x = MpiTracefile._instance

        assert isinstance(x, MpiTracefile)
        return x

    def __iadd__(self, entry):
        entry.pid = self.rank               
        self.__tracefile.append(entry)
        # OMG - I got so burnt with returning __tracefile. How insane.
        self.__tracefile.flush() # I think we need this to chase up some nasty lockup problems.
        return self
    
    append = __iadd__

    def add_metadata(self, 
                     pid=None,tid=None,
                     process_name=None,
                     process_labels=None,
                     process_sort_index=None,
                     thread_name=None,
                     thread_sort_index=None):
        if pid is None:
            pid = self.rank

        return self.__tracefile.add_metadata(pid, tid, process_name, process_labels, process_sort_index, thread_name, thread_sort_index)
    
    def now_ts(self):
        return self.__tracefile.now_ts()

    def close(self):
        if self.__tracefile is not None:
            self.__tracefile.close()







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
