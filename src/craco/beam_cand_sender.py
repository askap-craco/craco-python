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
from craco.mpi_cand_pipeline import *

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

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
    mydata = [my_comm.Get_rank()*3]
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbeams = my_comm.Get_size()
    
    destrank = MPI.COMM_WORLD.Get_rank()+nbeams+1

    assert destrank >= 0
    print(f'Sending from {rank} to {destrank}')
    comm.send(mydata, dest=destrank)

if __name__ == '__main__':
    _main()
