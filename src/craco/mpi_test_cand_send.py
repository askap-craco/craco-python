#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import time
from craco.mpi_candidate_buffer import MpiCandidateBuffer
from craco.mpiutil import np2array

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    values = parser.parse_args()

    # Run with mpirun -np 10 -hostfile $HOSTFILE  -x UCX_NET_DEVICES -x UCX_TLS -x UCX_IB_GID_INDEX python -m mpi4py src/craco/mpi_test_cand_gather.py 

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    nranks = world.Get_size()
    #cbuf = MpiCandidateBuffer()
    #cands = cbuf.cands
    if rank == 0:
        cbuf = MpiCandidateBuffer.for_tx(world, 1)
        cands = cbuf.cands
        ncand = 3
        cands['snr'][:ncand] = np.arange(ncand)
        cbuf.send(3)
    else:
        cbuf = MpiCandidateBuffer.for_rx(world, 0)
        cands = cbuf.cands
        ncand = cbuf.recv()
        #ncand = status.Get_elements(cbuf.mpi_dtype)
        print(f'Received cands ncand={ncand}')
        print(cands[:5])
              



if __name__ == '__main__':
    _main()
