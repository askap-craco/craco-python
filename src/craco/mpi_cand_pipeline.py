#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib

import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from IPython import embed

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def print_stuff():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    #embed()
    appnum = MPI.COMM_WORLD.Get_attr(MPI.APPNUM)
    my_app_comm = comm.Split(appnum, rank)
    print(f'{sys.argv[0]} rank {rank} of {numprocs} appnum={appnum} in my app is {my_app_comm.Get_rank()}/{my_app_comm.Get_size()}')
    return my_app_comm


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    my_comm = print_stuff()

    mydata = [my_comm.Get_rank()*2]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if my_comm.Get_rank() != 0:
        mydata = comm.recv()
        print(f'{my_comm.Get_rank()} got data from recv {mydata}')
    else:
        mydata = None
    
    results = my_comm.gather(mydata)
    print(f'Rank {my_comm.Get_rank()} got {results}')
    

if __name__ == '__main__':
    _main()
