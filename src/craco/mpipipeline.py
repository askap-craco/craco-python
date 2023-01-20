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
from craco import cardcap
from craco.card_averager import Averager
from craco.cardcapmerger import CcapMerger
import numba
from numba.typed import List

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


class CardFileSource:
    def __init__(self, cardidx, context, values):
        icard = cardidx % len(values.card)
        iblk = cardidx  // len(values.card)
        card = values.card[icard]
        block = values.block[iblk]
        fstr = f'b{block:02d}_c{card:02d}'
        myfiles = sorted(filter(values.files, lambda f: fstr in f))
        log.info(f'Cardidx {cardidx} icard={icard} iblk={iblk} card={card} blk={blk} has {len(myfiles)} files')
        assert myfiles > 0, f'No files for cardidx={cardidx} {fstr}'
        assert myfiles <= NFPGA, 'Too many files for cardidx={cardidx} {fstr}'
        self.merger = CcapMerger(myfiles)

    def packet_iter(self):
        return self.merger.packet_iter()

class Transposer:
    def __init__(self, dtype):
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        mpi_dtype = self.dtype
        s_msg = [dtx, (np2array(self.tx_counts), np2array(self.tx_displacements)), mpi_dtype]
        r_msg = [self.drx, (np2array(self.rx_counts), np2array(self.rx_displacements)), mpi_dtype]
        t_start = MPI.Wtime()
        comm.Alltoallv(s_msg, r_msg)
        t_end = MPI.Wtime()
        latency = (t_end - t_start)*1e6
        if rank == 0:
            print(f'Latency = {latency}us')

        return self.drx

class TransposeSender(Transposer):
    def __init__(self, cardidx, context, values):
        super().__init__(cardidx, context, values)
        nbeam = self.nbeam
        self.tx_counts[:nbeam] = 1
        self.tx_displacements[:nbeam] = np.arange(nbeam, dtype=np.int32))
        self.drx = np.zeros((1), dtype=self.dtype) # Dummy for all toall make zero if possible

    def send(self, dtx):
        return self.all2all(dtx)

class TransposeReceiver(Transposer):
    def __init__(self, cardidx, context, values):
        super().__init__(cardidx, context, values)
        nbeam = self.nbeam
        nrx = self.nrx
        self.rx_counts[nbeam:] = 1 # receive same amount from every tx
        self.rx_displacements[nbeam:] = np.arange(nrx, dtype=np.int32)
        self.drx = np.zeros(nrx, dtype=self.dtype)
        self.dtx = np.zeros((1), dtype=self.dtype) # dummy buffer for sending TODO: make zero if possible

    def recv(self):
        return self.all2all(self.dtx)

def proc_rx(cardidx, context, values):
    '''
    Process 1 card per beam
    1 card = 6 FGPAs
    '''

    nrx = values.nrx
    nbeam = values.nbeam
    nt = values.nt
    nant = 30
    nc = NCHAN*NFPGA
    npol = 1 if values.pol_sum else 2

    assert numprocs == nrx + nbeam
    avg = Averager(nbeam, nant, nc, nt, npol)
    #ccap = CardCapturer(values, pvcache=context)
    ccap = CardFileSource(cardidx, context, values)
    transposer = TransposeSender(cardidx, context, values)
    # construct a typed list for numba - it's a bit of a pain but it needs to be done this way
    # Just need some types of the data so the list can be typed
    # data = List()
    #[data.append(fcap.rdma_buffers[0][0]) for fcap in ccap.fpga_cap]

    for ibuf, packets in ccap.packet_iter():
        averaged = averager.accumulate_all(packets)
        transposer.send(averaged)

def proc_beam(beamid, context, values):
    nrx = values.nrx
    nbeam = values.nbeam
    nt = values.nt

    assert numprocs == nrx + nbeam
    transposer = TransposeReceiver(beamid, context, values)
    cas_filterbank = None
    ics_filterbank = None
    vis = None
    while True:
        beam_data = tranposer.recv()
        cas_filterbank.write(beam_data['cas'])
        ics_filterbank.write(beam_data['cas'])
        vis.write(beam_data['vis'])


    
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
    parser.add_arguments('--nbeams', type=int, help='Number of beams', default=36)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    if values.dump_rankfile:
        dump_rankfile(values)
        sys.exit(0)

    # beams first, then rx
    if rank < values.nbeam:
        proc_beam(rank, values)
    else:
        proc_rx(rank - values.nbeam, values)
                  
    

if __name__ == '__main__':
    _main()
