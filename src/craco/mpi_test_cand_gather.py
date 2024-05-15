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
    
    if rank == 0:
        cands = MpiCandidateBuffer.for_beam_manager(world)
        cands.gather()    
        print(cands.cands)
    else:
        cands = MpiCandidateBuffer.for_beam_processor(world)
        cands.cands['snr'] = rank*10 + np.arange(len(cands.cands))
        #print(cands.cands)

        cands.gather()


if __name__ == '__main__':
    _main()
