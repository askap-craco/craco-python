#!/usr/bin/env python
"""
MPI Utility classe

Copyright (C) CSIRO 2020
"""
import mpi4py
from mpi4py import MPI
import logging
from array import array
import numpy as np

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)

def np2array(d):
    '''
    This is how :https://github.com/erdc/mpi4py/blob/master/demo/osu_alltoallv.py does it
    Not sure it's absolutely necessary
    '''
    assert d.dtype == np.int32
    a =  array('i', d)
    
    return a


def parse_hostfile(hostfile):
    '''
    parse a host file
    A list of host files.
    The slots=X is ignored
    '''
    
    hosts = []
    with open(hostfile, 'r') as hf:
        for line in hf:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            
            bits = line.split()
            hosts.append(bits[0])

    return hosts



import platform
FORMAT = '%(asctime)s [%(hostname)s:%(process)d] r%(rank)d %(module)s %(message)s'
class HostnameFilter(logging.Filter):
    hostname = platform.node()
    def __init__(self, comm):
        self.rank =  comm.Get_rank()
    
    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        record.rank = self.rank
        return True

def make_log_handler(comm):
    handler = logging.StreamHandler()
    handler.addFilter(HostnameFilter(comm))
    handler.setFormatter(logging.Formatter(FORMAT))
    return handler

def setup_logging(comm, verbose=False):
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger() # get root logger
    logger.addHandler(make_log_handler(comm))
    logger.setLevel(level)


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
