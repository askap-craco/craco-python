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

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class MpiApp:
    def __init__(self):
        self.world = MPI.COMM_WORLD
        comm = self.world
        self.world_rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.world_comm = comm
    
        appnum = MPI.COMM_WORLD.Get_attr(MPI.APPNUM)        
        self.app_num = appnum

        # Communicator for my app
        self.app_comm = comm.Split(appnum, self.world_rank)
        self.app_rank = self.app_comm.Get_rank()
        self.app_size = self.app_comm.Get_size()

    def __str__(self):
        s = f'world {self.world_rank}/{self.world_size} app={self.app_num} {self.app_rank}/{self.app_size}'
        return s

def do_all2all(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    senddata = (rank+1)*np.arange(size, dtype=int)
    recvdata = np.empty(size, dtype=int)
    comm.Alltoall(senddata, recvdata)
    return recvdata

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

    app = MpiApp()
    world_data = do_all2all(app.world)
    print(f'{app} World results {world_data}')

    app_data = do_all2all(app.app_comm)
    print(f'{app} App results {app_data}')
    
    if app.world_rank == 0:
        time.sleep(10)
        req = app.world.isend('test', dest=1)
        req.wait()
    elif app.world_rank == 1:
        req = app.world.irecv(source=0)
        finished = False
        while not finished:
            finished, data = req.test()
            print(f'Waiting  {finished} {data}')
            time.sleep(1)


    
    

if __name__ == '__main__':
    _main()
