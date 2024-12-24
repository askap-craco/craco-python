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
import datetime
from craco.mpi_ring_buffer_manager import MpiRingBufferManager
from craco.visblock_accumulator import VisblockAccumulatorStruct

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
world_comm = MPI.COMM_WORLD
world_rank = world_comm.rank


def simple_test():
    '''
    Copied from:
    https://gist.github.com/rcthomas/d1f0ad0f71a61d791b7e5d96ab259559
    '''
    world_comm = MPI.COMM_WORLD
    node_comm  = world_comm.Split_type(MPI.COMM_TYPE_SHARED)

    size = 1000
    disp_unit = MPI.DOUBLE.Get_size()
    win = MPI.Win.Allocate_shared(size * disp_unit if node_comm.rank == 0 else 0,
            disp_unit, comm = node_comm)

    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPI.DOUBLE.Get_size()
    buf = np.array(buf, dtype='B', copy=False)
    ary = np.ndarray(buffer=buf, dtype='d', shape=(size,))

    if node_comm.rank == 1:
        ary[:5] = np.arange(5)
    node_comm.Barrier()
    if node_comm.rank == 0:
        print(ary[:10])

def test_lock_all():
    '''
    Annoyting -  Local_all() doesnt stop other people from reading/weriting.
    It doesnt block
    '''
    print('****8test lock all****')
    world_comm = MPI.COMM_WORLD
    node_comm  = world_comm # world_comm.Split_type(MPI.COMM_TYPE_SHARED)

    size = int(10e6)
    disp_unit = MPI.DOUBLE.Get_size()
    win = MPI.Win.Allocate_shared(size * disp_unit if node_comm.rank == 0 else 0,
            disp_unit, comm = node_comm)

    buf, itemsize = win.Shared_query(0)
    assert itemsize == MPI.DOUBLE.Get_size()
    buf = np.array(buf, dtype='B', copy=False)
    ary = np.ndarray(buffer=buf, dtype='d', shape=(size,))
    rank = node_comm.rank

    if node_comm.rank == 1:
        print(f'r{rank} locking')
        win.Lock_all()

    print(f'r{rank} barrier')
    node_comm.Barrier()

    if node_comm.rank == 1:
        ary[:5] = np.arange(5)
        print(f'r{rank} wrote memory. Sleeping....')
        time.sleep(5)
        print(f'r{rank} unlocking')
        win.Unlock_all()
    #node_comm.Barrier()
    if node_comm.rank == 0:
        print(f'r{rank} locking')
        win.Lock_all()
        print(f'r{rank} locked. Data is', ary[:10])
        print(f'r{rank} unlocking')
        win.Unlock_all()

def myprint(s):
    now = datetime.datetime.now().isoformat()
    print(now, f'r{world_rank}', s)

def test_ringbuffer():
    world_comm = MPI.COMM_WORLD
    rank = world_comm.rank
    nslots = 10
    rb = MpiRingBufferManager(world_comm, nslots, tx_rank =0, rx_rank=1)
    nbl = 200
    nf = 288
    nt = 256
    ra = VisblockAccumulatorStruct(nbl, nf, nt, world_comm, nslots)
    # THIS test will overflow the buffer - that's what we want
    for x in range(10):
        if rank == 0:
            i = rb.open_write()
            myprint(f'writing {i}')
            ra.pipeline_data_array[i]['bl_weights'][:] = i*2
            time.sleep(1)
            rb.close_write()
            myprint('Closed')
        else:
            myprint(f'Waiting to read')
            i = rb.open_read()
            bl = ra.pipeline_data_array[i]['bl_weights'][0]
            myprint(f'Reading slot={i} got={bl}')
            
            time.sleep(2)
            myprint(f'closing {i}')
            rb.close_read()

        






def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    #parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    #simple_test()
    #test_lock_all()
    test_ringbuffer()

    


    

if __name__ == '__main__':
    _main()
