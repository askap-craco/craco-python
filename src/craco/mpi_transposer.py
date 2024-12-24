#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import craco.card_averager
from craco.timer import Timer


import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class CustomMpiTransposer:
    def __init__(self, comm:MPI.Comm, nbeam, nrx, dtype):            
        self.comm = comm
        self.rank = comm.Get_rank()
        size = comm.Get_size()
        assert nbeam + nrx == size, f'Invalid size. nbeam={nbeam} nrx={nrx} size={size}'
        self.nbeam = nbeam
        self.nrx = nrx
        self.requests = []
        self.dtype = dtype
        self.mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.dtype)
        is_rx = self.rank < self.nrx
        is_beam = not is_rx
        colour = 0 if is_rx else 1
        self.my_comm = comm.Split(colour)


    def rank_of_beam(self, ibeam):
        '''
        Ranks are ordered 0->nrx-1 | nrx -> nrx + nbeam-1
        '''
        return self.nrx + ibeam

    def rank_of_rx(self, irx):
        return irx

    def _post(self, req):
        self.requests.append(req)

    def wait(self):
        for r in self.requests:
            r.wait()
        
        self.requests.clear()



class MpiTransposeSender(CustomMpiTransposer):

    def __init__(self, comm, nbeam, nrx, dtype):            
        super().__init__(comm, nbeam, nrx, dtype)
        lut = []
        my_rx = self.rank

        for ibeam in range(self.nbeam):
            for irx in range(self.nrx):            
                destbeam = (ibeam + irx) % self.nbeam
                


    def send(self, dtx, do_async=False):
        '''
        do-async=True tends to add a lot more jitter
        '''
        assert len(dtx) == self.nbeam
        t = Timer()
        # synchronise the senders # so we don't get out of step and pile onto a receiver by accident
        self.my_comm.Barrier()
        t.tick('Barrier')
        for ibeam in range(self.nbeam):
            # make sure everyone doesn't send to the same place
            # at once
            destbeam = (ibeam + self.rank) % self.nbeam
            destrank = self.rank_of_beam(destbeam)
            #print(f'r{self.rank} sending to r{destrank} beam={destbeam}')
            if do_async:
                self._post(self.comm.Isend([dtx[destbeam:destbeam+1], self.mpi_dtype], dest=destrank))
                t.tick('Isend')
            else:
                self.comm.Ssend([dtx[destbeam:destbeam+1], self.mpi_dtype], dest=destrank)
                t.tick('send', args={'destbeam':destbeam, 'destrank':destrank})
            
            #print(f'r{self.rank} sent to r{destrank} beam={destbeam}')

    def send_with_barrier(self, dtx):
        assert len(dtx) == self.nbeam
        t = Timer()
        ibeam = 0
        for irx in range(self.nrx):
            # make sure everyone doesn't send to the same place
            # at once
            destbeam = (ibeam + self.rank) % self.nbeam
            destrank = self.rank_of_beam(destbeam)
            if True:
                self.comm.Send([dtx[destbeam:destbeam+1], self.mpi_dtype], dest=destrank)
                t.tick('send', args={'destbeam':destbeam, 'destrank':destrank})
                ibeam += 1
            
            self.my_comm.Barrier()



class MpiTransposeReceiver(CustomMpiTransposer):

    def __init__(self, comm, nbeam, nrx, dtype):            
        super().__init__(comm, nbeam, nrx, dtype)
        my_beamid = self.rank - self.nrx
        assert 0 <= my_beamid < self.nbeam, f'Invalid beamid {my_beamid}'
        lut = []

        for irx in range(self.nrx):            
          for ibeam in range(self.nbeam):
                destbeam = (ibeam + irx) % self.nbeam
                if destbeam == my_beamid:
                    lut.append(irx)

        if my_beamid == 0:
            print(lut)
        self.lut = lut        

    def Irecv(self, drx, do_async=False):
        assert len(drx) == self.nrx
        my_beamid = self.rank - self.nrx
        assert my_beamid >= 0
        t = Timer()
  
        
        for i in range(self.nrx):
            #sourcerx = (my_beamid - irx) % self.nrx # this is functionally incorrect, but super fast
            irx = self.lut[i]
            sourcerx = irx
            sourcerank = self.rank_of_rx(sourcerx)            
            #print(f'r{self.rank} waiting for r{sourcerank} rx={sourcerx}')
            #print(drx[irx].dtype, drx[irx].shape, self.dtype, self.mpi_dtype)
            # Need to give it a length=1 array, otherwise it complains it's got 
            # a scalar and .... complains.
            if do_async:
                self._post(self.comm.Irecv([drx[irx:irx+1], self.mpi_dtype], source=sourcerank))
                t.tick('Irecv', args={'sourcerx':irx, 'sourcerank':sourcerank})
            else:
                self.comm.Recv([drx[irx:irx+1], self.mpi_dtype], source=sourcerank)
                t.tick('recv', args={'sourcerx':irx, 'sourcerank':sourcerank})
            
            #print(f'r{self.rank} recieved for r{sourcerank} rx={sourcerx}')

        return self.requests


    def recv(self, drx):
        self.Irecv(drx)
        self.wait()




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


    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()

    nbeam = 36
    nant = 24
    nc = 24
    nt = 32
    npol = 1
    vis_fscrunch = 6
    vis_tscrunch = 1
    real_dtype = np.float32
    cplx_dtype = np.float32

    nrx = nranks - nbeam
    assert nrx > 0
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, real_dtype, cplx_dtype)
    vis_nc = nc // vis_fscrunch
    nprocs = nrx + nbeam
    assert nranks  == nprocs, f'Expected {nprocs} but got size {nranks}'
    rank = comm.Get_rank()

    if rank == 0:
        print(f'Dtype is {dt}')

    niter = 50
    
    comm.Barrier() # get everyone talking

    # check how long barriers take
    t = Timer()
    for i in range(20):
        comm.Barrier()
        t.tick('barrier', args={'i':i})

    if rank < nrx:
        sender = MpiTransposeSender(comm, nbeam, nrx, dt)
        if rank == 0:
            print(sender.dtype, sender.mpi_dtype)
        dtx = np.zeros((nbeam), dtype=dt)
        irx = rank
        for ibeam in range(nbeam):
            dtx[ibeam]['ics'][:] = ibeam
            dtx[ibeam]['cas'][0,:] = np.arange(nc) + irx*nc

        sender.send(dtx)
        comm.Barrier()
        t_start = MPI.Wtime()
        for i in range(niter):
            t = Timer()
            comm.Barrier()
            t.tick('Barrier 1')
            sender.send(dtx)
            t.tick('send')
            comm.Barrier()
            t.tick('Barrier 2')
        t_end = MPI.Wtime()
    else:
        receiver = MpiTransposeReceiver(comm, nbeam, nrx, dt)
        drx = np.zeros((nrx), dtype=dt)        
        receiver.recv(drx)

        comm.Barrier()
        t_start = MPI.Wtime()       
        for i in range(niter):
            t = Timer()
            comm.Barrier()
            t.tick('Barrier 1')            
            receiver.Irecv(drx)
            t.tick('IRecv')
            receiver.wait()
            t.tick('wait')
            comm.Barrier()
            t.tick('Barrier2')
        t_end = MPI.Wtime()
        latency = (t_end - t_start)/niter

        #print(f'r{rank} latency', latency)

        ibeam = rank - nrx
        firstd = drx['ics'].flat[0]
        assert np.all(drx['ics'] == firstd), f'ICS not equal to {firstd}'
        assert np.all(drx['ics'] == ibeam), f'ICS not equal to {ibeam}'
        #np.save(f'drx_b{ibeam:02d}', drx, allow_pickle=True)
        chans = drx['cas'][:,0,:].flatten()
        #print(f'ibeam={ibeam} c[0]={chans[0]} clast={chans[-1]} cmin={chans.min()}  cmax={chans.max()} chans={chans}')
        assert np.all(chans == np.arange(nrx*nc))
        #print('Data checks sucessfull')
        
    latency = (t_end - t_start)/niter

    max_latency = comm.reduce(latency, MPI.MAX)
    min_latency = comm.reduce(latency, MPI.MIN)
    sum_latency = comm.reduce(latency, MPI.SUM) 
    if rank == 0:
        mean_latency = sum_latency / nranks
        rx_size_bytes = dt.itemsize * nrx
        rx_gbps = rx_size_bytes * 8 / mean_latency / 1e9
        print(f'LATENCY SUMMARY: {max_latency*1e3:0.2f}/{min_latency*1e3:0.2f}/{mean_latency*1e3:0.1f} ms mean={rx_gbps:0.1f} Gbps nrx={nrx} rxsize={rx_size_bytes/1e6} MB' )
    

if __name__ == '__main__':
    _main()
