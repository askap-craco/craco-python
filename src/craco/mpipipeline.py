#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from array import array


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

nrx = 144
nbeam = 36
nt = 16
ncperrx = 4

assert numprocs == nrx + nbeam

# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]
dtype=np.int32
mpi_dtype=MPI.INT32_T
#dtype=np.uint8
#mpi_dtype=MPI.BYTE

def np2array(d):
    '''
    This is how :https://github.com/erdc/mpi4py/blob/master/demo/osu_alltoallv.py does it
    '''
    assert d.dtype == np.int32
    a =  array('i', d)
    
    return a

def myalltoall(comm, dtx, tx_counts, tx_displacements, drx, rx_counts, rx_displacements):
    s_msg = [dtx, (np2array(tx_counts), np2array(tx_displacements)), mpi_dtype]
    r_msg = [drx, (np2array(rx_counts), np2array(rx_displacements)), mpi_dtype]
    print('rank', rank, 'RX', drx.size, rx_counts, rx_displacements, 'TX', dtx.size, tx_counts, tx_displacements)
    comm.Alltoallv(s_msg, r_msg)

    #size = dtx.size // numprocs
    #comm.Alltoall([dtx, size, MPI.INT], [drx, size, MPI.INT])

def proc_rx(chanid, values):
    # need beams on the outer
    dtx = np.arange(nbeam*nt*ncperrx, dtype=dtype).reshape((nbeam, ncperrx,nt)) + chanid
    drx = np.zeros_like(dtx) # should be zero at the end - ideally never allocated or written

    tx_counts = np.zeros(numprocs, np.int32)
    tx_displacements = np.zeros(numprocs, np.int32)
    rx_counts = np.zeros(numprocs, np.int32)
    rx_displacements = np.zeros(numprocs, np.int32)
    
    tx_counts[:nbeam] = nt*ncperrx # send same amount to every rx
    tx_displacements[:nbeam] = np.arange(nbeam)*nt*ncperrx
    
    myalltoall(comm, dtx, tx_counts, tx_displacements, drx, rx_counts, rx_displacements)

def proc_beam(beamid, values):
    drx = np.zeros(nt*nrx*ncperrx, dtype=dtype).reshape(nrx*ncperrx, nt)
    dtx = np.zeros_like(drx)
    tx_counts = np.zeros(numprocs, np.int32)
    tx_displacements = np.zeros(numprocs, np.int32)
    rx_counts = np.zeros(numprocs, np.int32)
    rx_displacements = np.zeros(numprocs, np.int32)
    
    rx_counts[nbeam:] = nt*ncperrx # receive same amount from every tx
    rx_displacements[nbeam:] = np.arange(nrx)*nt*ncperrx

    myalltoall(comm, dtx, tx_counts, tx_displacements, drx, rx_counts, rx_displacements)
    print('rank', rank, drx)
    

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

    # beams first, then rx
    if rank < nbeam:
        proc_beam(rank, values)
    else:
        proc_rx(rank - nbeam, values)
                  
    

if __name__ == '__main__':
    _main()
