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
from craft import craco_plan
from craft import uvfits
import time

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
    if rank == 0:
        log.info('Loading UV coordinates from file %s ', values.uv)
        f = uvfits.open(values.uv)
        plan = craco_plan.PipelinePlan(f, values)
        plan.fdmt_plan
        log.info('sending plan %s', plan)
        world.send(plan, dest=1)
    else:
        #plan = world.recv(source=0)
        size = 16*1024*1024
        req = world.irecv(size, source=0)

        while True:
            log.info('testing')
            ok, plan = req.test()
            log.info('Test got %s', ok)
            if ok:
                break
            log.info('Sleeping')
            time.sleep(1)
        log.info('Got plan! %s', plan)


    

if __name__ == '__main__':
    _main()
