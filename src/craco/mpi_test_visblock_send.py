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
from craft import craco_plan
from craft import uvfits
from craco.timer import Timer
import time
from craco.visblock_accumulator import VisblockAccumulatorStruct

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter, parents=[craco_plan.get_parser()], conflict_handler='resolve')
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    nbl = 435
    nc = 288
    nt = 256
    va = VisblockAccumulatorStruct(nbl,nc,nt)
    mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(va.pipeline_data_array.dtype)
    msg = [va.pipeline_data_array, va.pipeline_data_array.size, mpi_dtype]

    t = Timer()
    if rank == 0:        
        log.info('Sending')
        world.Send(msg, dest=1)
        log.info('sent')
        t.tick('Send')
    else:
        log.info('receving')
        world.Recv(msg, source=0)
        log.info('received')
        t.tick('Recv')

    print(t)

    


        

    

if __name__ == '__main__':
    _main()
