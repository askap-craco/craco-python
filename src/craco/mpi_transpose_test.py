#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco import cardcap
from craco.mpiutil import np2array


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()
ncperrx = 4


# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]
dtype=np.int32
mpi_dtype=MPI.INT32_T
#dtype=np.uint8
#mpi_dtype=MPI.BYTE


def myalltoall(comm, dtx, tx_counts, tx_displacements, drx, rx_counts, rx_displacements):
    s_msg = [dtx, (np2array(tx_counts), np2array(tx_displacements)), mpi_dtype]
    r_msg = [drx, (np2array(rx_counts), np2array(rx_displacements)), mpi_dtype]
    #print('rank', rank, 'RX', drx.size, rx_counts, rx_displacements, 'TX', dtx.size, tx_counts, tx_displacements)
    t_start = MPI.Wtime()
    comm.Alltoallv(s_msg, r_msg)
    t_end = MPI.Wtime()
    latency = (t_end - t_start)*1e6
    if rank == 0:
        print(f'Latency = {latency}us')
        
    #size = dtx.size // numprocs
    #comm.Alltoall([dtx, size, MPI.INT], [drx, size, MPI.INT])

def proc_rx(chanid, values):

    nrx = values.nrx
    nbeam = values.nbeam
    nt = values.nt

    assert numprocs == nrx + nbeam

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
    nrx = values.nrx
    nbeam = values.nbeam
    nt = values.nt

    assert numprocs == nrx + nbeam

    drx = np.zeros(nt*nrx*ncperrx, dtype=dtype).reshape(nrx*ncperrx, nt)
    dtx = np.zeros_like(drx)
    tx_counts = np.zeros(numprocs, np.int32)
    tx_displacements = np.zeros(numprocs, np.int32)
    rx_counts = np.zeros(numprocs, np.int32)
    rx_displacements = np.zeros(numprocs, np.int32)
    
    rx_counts[nbeam:] = nt*ncperrx # receive same amount from every tx
    rx_displacements[nbeam:] = np.arange(nrx)*nt*ncperrx

    myalltoall(comm, dtx, tx_counts, tx_displacements, drx, rx_counts, rx_displacements)

    
def dump_rankfile(values):
    from craco import mpiutil
    hosts = mpiutil.parse_hostfile(values.hostfile)
    log.debug("Hosts %s", hosts)
    nrx = len(values.block)*len(values.card)*len(values.fpga)
    nbeams = values.nbeams
    nranks = nrx + nbeams
    total_cards = len(values.block)*len(values.card)
    ncards_per_host = (total_cards + len(hosts))//len(hosts)
    nrx_per_host = ncards_per_host*6
    nbeams_per_host = (nbeams + len(hosts))//len(hosts)
    log.info(f'Spreading {nranks} over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas and {nbeams} beams with {nbeams_per_host} per host')

    rank = 0
    with open(values.dump_rankfile, 'w') as fout:
        # add all the beam processes
        for beam in range(nbeams):
            hostidx = rank // nbeams_per_host
            hostrank = rank % nbeams_per_host
            host = hosts[hostidx]
            slot = 0 # put on the U280 slot
            core='1-40'
            s = f'rank {rank}={host} slot={slot}:{core} # Beam {beam}\n'
            fout.write(s)
            rank += 1

        rxrank = 0
        for block in values.block:
            for card in values.card:
                for fpga in values.fpga:
                    hostidx = rxrank // nrx_per_host
                    hostrank = rxrank % nrx_per_host
                    host = hosts[hostidx]
                    slot = 1 # fixed because both cards are on NUMA=1
                    # Put different FPGAs on differnt cores
                    evenfpga = fpga % 2 == 0
                    core = rank % 10
                    slot = 1
                    s = f'rank {rank}={host} slot={slot}:{core} # Block {block} card {card} fpga {fpga}\n'
                    fout.write(s)
                    rank += 1
                    rxrank += 1


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    cardcap.add_arguments(parser)
    parser.add_argument('--nbeams', type=int, help='Number of beams', default=36)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    if values.dump_rankfile:
        dump_rankfile(values)
        sys.exit(0)

    print(values.block, values.card, values.fpga)
    values.nrx = len(values.block)*len(values.card)*len(values.fpga)//6

    # beams first, then rx
    if rank < values.nbeams:
        proc_beam(rank, values)
    else:
        proc_rx(rank - values.nbeam, values)
                  
    

if __name__ == '__main__':
    _main()
