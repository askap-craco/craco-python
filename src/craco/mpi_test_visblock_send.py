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
from craco.visblock_accumulator import VisblockAccumulatorStruct,SharedVisblockSender

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
    nt =  256
    va = VisblockAccumulatorStruct(nbl,nc,nt)
    msg = va.mpi_msg
    t = Timer()

    
    if rank == 0:        
        log.info('Sending')
        req = world.Isend(msg, dest=1)
        log.info('sent')        
        t.tick('Send')
        req.wait()
        t.tick('send complete')
    else:
        log.info('receving')
        req = world.Irecv(msg, source=0)
        log.info('received')        
        t.tick('IRecv')
        req.wait()
        t.tick('Irecv complete')

    print(t)
    size = 1
    disp_unit = va.dtype.itemsize
    node_comm = world
    win = MPI.Win.Allocate_shared(size * disp_unit if world.rank == 0 else 0,
            disp_unit, comm = node_comm)

    buf, itemsize = win.Shared_query(0)
    assert itemsize == va.dtype.itemsize
    #buf = np.array(buf, dtype=va.dtype, copy=False)
    ary = np.ndarray(buffer=buf, dtype=va.dtype, shape=(size,))
    d = va.pipeline_data
    d['bl_weights'] = np.arange(nbl)
    d['tf_weights'] = np.arange(nc*nt).reshape(d['tf_weights'].shape)
    d['vis'].real = np.arange(nc*nt*nbl).reshape(d['vis'].shape)
    
    t = Timer()
    if rank == 0:                
        ary['bl_weights'] = d['bl_weights']
        ary['tf_weights'] = d['tf_weights']
        ary['vis'] = d['vis']
        t.tick('setting')
        log.info('Sending')
        req = world.send(0, dest=1)
        log.info('sent')        
        t.tick('Send')        
    else:
        iblk = world.recv(source=0)
        log.info(f'received {iblk}')        
        t.tick('recv')
        np.testing.assert_equal(d, ary)
        t.tick('Compare')
        d['bl_weights'] = ary['bl_weights']
        d['tf_weights'] = ary['tf_weights']
        d['vis'] = ary['vis']
        t.tick('copy')


    print(t)

    # test newer version

    vsend = SharedVisblockSender(nbl,nc,nt, world, 3)
    if rank == 0:
        for i in range(3):
            vsend.va.pipeline_data_array[i]['bl_weights'] = i

        
        
        


    


        

    

if __name__ == '__main__':
    _main()
