#!/usr/bin/env python
import sys
import logging
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import numpy as np
import timeit
import time



def _main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    print(f'My rank is {rank}/{numprocs}')
    nbeam = numprocs
    nant = 30
    nbl = nant*(nant - 1)//2
    nc = 15
    nt = 128*nc
    nt = nc

    din = np.zeros((nbeam, nbl*nt), dtype=np.int32)
    din.flat = np.arange(din.size)*(rank+1)
    print(f'Shape {din.shape} {din.nbytes} bytes')

    # outputs is all channels for 1 beam - but we assume 36 beams and 36 channels
    dout = np.zeros_like(din)
    size = nbl*nt # number of entries to zend
    
    send_msg = [din, size, MPI.INT]
    recv_msg = [dout, size, MPI.INT]
    niter = 100
    total_duration = 0
    print(f'Input for {rank} is {din.shape}')
    for iter in range(niter):
        start = time.time()
        comm.Alltoall(send_msg, recv_msg)
        end = time.time()
        duration = end - start
        total_duration += duration
        rate = (dout.nbytes*8)/duration/1e9
        if rank == 0:
            print(f'Exectime = {duration} {rate} Gbps')

    total_duration /= niter

    print(f'output for {rank} is {dout.shape}')
    if rank == 0:
        rate = (dout.nbytes*8)/total_duration/1e9
        print(f'Total Exectime = {duration} {rate} Gbps')

    
    

if __name__ == '__main__':
    _main()
