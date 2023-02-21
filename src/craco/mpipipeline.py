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
from craco.cardcap import NCHAN, NFPGA
import craco.card_averager
from craco.card_averager import Averager
from craco.cardcapmerger import CcapMerger
from craco.mpiutil import np2array
from craco.vissource import VisSource
import numba
import glob
import craft.sigproc as sigproc

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]

REAL_DTYPE = np.float32 # Transpose/averaging type for real types
CPLX_DTYPE = np.complex64 # Tranaspose/averagign dtype for complex types  can also be a real type, like np.int16 and somethign willl add an extra dimension


def get_transpose_dtype(values):
    nbeam = values.nbeams
    nt = values.nt
    nant = 30
    nc = NCHAN*NFPGA
    vis_fscrunch = values.vis_fscrunch
    vis_tscrunch = values.vis_tscrunch
    npol = 1 if values.pol_sum else 2
    rdtype = np.float32
    cdtyep = np.complex64
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, REAL_DTYPE, CPLX_DTYPE)
    return dt

## OK KB - YOU NEED TO TEST THE AVERAGER WITH THE INPUT DATA

class CardFileSource:
    '''
    Gets data from a single card - i.e. 6 FPGAs
    '''
    def __init__(self, cardidx, vis_source, values):
        icard = cardidx % len(values.card)
        iblk = cardidx  // len(values.card)
        card = values.card[icard]
        block = values.block[iblk]
        if values.cardcap_dir is not None:
            files = glob.glob(os.path.join(values.cardcap_dir, '*.fits'))
            log.debug('Found %d cards in %s', len(files), values.cardcap_dir)
        else:
            files = values.files
            
        fstr = f'b{block:02d}_c{card:02d}'
        myfiles = sorted(filter(lambda f: fstr in f, files))
        log.info(f'Cardidx {cardidx} icard={icard} iblk={iblk} card={card} iblk={iblk} has {len(myfiles)} files ')
        assert len(myfiles) == NFPGA, f'Incorrect number of cards for cardidx={cardidx} {fstr} {myfiles}'
        self.merger = CcapMerger(myfiles)

    def packet_iter(self):
        return self.merger.packet_iter()

class Transposer:
    def __init__(self, cardidx, vis_source, values):
        # OK - now this is where it gets tricky and i wish I could refactor it properlty to get what I'm expecting
        self.nbeam = values.nbeams
        self.nrx = values.nrx
        self.dtype = get_transpose_dtype(values)
        self.mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.dtype)
        
        self.cardidx = cardidx
        self.vis_source = vis_source
        self.values = values
        
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        s_msg = [dtx, (np2array(self.tx_counts), np2array(self.tx_displacements)), self.mpi_dtype]
        r_msg = [self.drx, (np2array(self.rx_counts), np2array(self.rx_displacements)), self.mpi_dtype]
        #print(f'all2all Rank {rank}/{numprocs} {self.dtype}={self.dtype.itemsize}bytes {self.mpi_dtype} TX:{self.tx_counts} {self.tx_displacements} RX:{self.rx_counts}-{self.rx_displacements}')

        t_start = MPI.Wtime()
        comm.Alltoallv(s_msg, r_msg)
        t_end = MPI.Wtime()
        latency = (t_end - t_start)*1e3
        if rank == 0:
            print(f'RANK0 Latency = {latency}ms')

        #print(f'all2all COMPLETE {rank}/{numprocs} latency={latency} ms {t_start} {t_end}')

        return self.drx

class TransposeSender(Transposer):
    def __init__(self, cardidx, vis_source, values):
        super().__init__(cardidx, vis_source, values)
        nbeam = self.nbeam
        self.tx_counts[:nbeam] = 1
        self.tx_displacements[:nbeam] = np.arange(nbeam, dtype=np.int32)
        self.drx = np.zeros((1), dtype=self.dtype) # Dummy for all toall make zero if possible

    def send(self, dtx):
        assert len(dtx) == self.nbeam
        return self.all2all(dtx)

class TransposeReceiver(Transposer):
    def __init__(self, cardidx, vis_source, values):
        super().__init__(cardidx, vis_source, values)
        nbeam = self.nbeam
        nrx = self.nrx
        self.rx_counts[nbeam:] = 1 # receive same amount from every tx
        self.rx_displacements[nbeam:] = np.arange(nrx, dtype=np.int32)
        self.drx = np.zeros(nrx, dtype=self.dtype)
        self.dtx = np.zeros((1), dtype=self.dtype) # dummy buffer for sending TODO: make zero if possible

    def recv(self):
        return self.all2all(self.dtx)


def proc_rx(cardidx, vis_source, values):
    '''
    Process 1 card per beam
    1 card = 6 FGPAs
    '''

    nrx = values.nrx
    nbeam = values.nbeams
    nt = values.nt
    nant = 30
    nc = NCHAN*NFPGA
    npol = 1 if values.pol_sum else 2

    assert numprocs == nrx + nbeam


    ccap = CardFileSource(cardidx, vis_source, values)
    dummy_packet = np.zeros((ccap.merger.npackets_per_frame), dtype=ccap.merger.dtype)
    averager = Averager(nbeam, nant, nc, nt, npol, values.vis_fscrunch, values.vis_tscrunch, REAL_DTYPE, CPLX_DTYPE, dummy_packet)
    transposer = TransposeSender(cardidx, vis_source, values)

    # construct a typed list for numba - it's a bit of a pain but it needs to be done this way
    # Just need some types of the data so the list can be typed
    
    t_start = MPI.Wtime()

    log.debug('Dummy packet shape %s dtype=%s', dummy_packet.shape, dummy_packet.dtype)
    # Run dummy packet into averager to make it compile
    dummy_packets = [dummy_packet for pkt in range(NFPGA)]
    #averager.accumulate_packets(dummy_packets)
    #averager.reset()
    
    for ibuf, (packets, fids) in enumerate(ccap.packet_iter()):
        now = MPI.Wtime()
        read_time = now - t_start
        # so we have to put a dummy value in and add a separate flags array
        avg_start = MPI.Wtime()
        averaged = averager.accumulate_packets(packets)
        avg_end = MPI.Wtime()
        avg_time = avg_end - avg_start
        #print('RX times', read_time, avg_time, t_start, now, avg_start, avg_end)
        test_mode = False
        if test_mode:
            averaged['ics'].flat = cardidx #np.arange(averaged['ics'].size)
            averaged['cas'].flat = np.arange(averaged['cas'].size)
            averaged['vis'].flat = np.arange(averaged['vis'].size)

        transposer.send(averaged)
        #averager.update_scales()
        averager.reset()
        t_start = MPI.Wtime()

class FilterbankSink:
    def __init__(self, prefix, beamid, vis_source, values):
        fname = os.path.join(values.outdir, f'{prefix}_b{beamid:02d}.fil')
        hdr = {'nbits':32,
               'nchans':vis_source.nchans,
               'nifs':vis_source.npol,
               'src_raj':vis_source.ra,
               'src_dej':vis_source.dec,
               'tstart':vis_source.tstart,
               'tsamp':vis_source.tsamp,
               'fch1':vis_source.fch1,
               'foff':vis_source.foff}

        self.fout = sigproc.SigprocFile(fname, 'wb', hdr)

    def write(self, beam_data):
        '''
        Writes data to the filterbank
        data has shape (nrx, nt,  nchan_per_rx) e.g. (36, 128, 24) 
        '''

        assert beam_data.dtype == np.float32, 'WE set nbits=32 for this filterbank, so if it isnt we need to do some work'
        dout = np.transpose(beam_data, [1, 0, 2])
        if rank == 0:
            print(dout.shape, dout.flatten().shape, dout.size, dout.mean(), dout.std())
        dout.tofile(self.fout.fin)

    def close(self):
        self.fout.fin.close()
        
        
def proc_beam(beamid, vis_source, values):
    nrx = values.nrx
    nbeam = values.nbeams
    nt = values.nt
    nant = 30
    nc = NCHAN*NFPGA
    npol = 1 if values.pol_sum else 2

    assert numprocs == nrx + nbeam, f'Invalid MPI setup numprocs={numprocs} nrx={nrx} nbeam={nbeam} expected {nrx + nbeam}'

    transposer = TransposeReceiver(beamid, vis_source, values)
    cas_filterbank = None
    ics_filterbank = None
    vis = None
    os.makedirs(values.outdir, exist_ok=True)

    cas_filterbank = FilterbankSink('cas', beamid, vis_source, values)
    ics_filterbank = FilterbankSink('ics', beamid, vis_source, values)

    
    while True:
        beam_data = transposer.recv()
        #@print(f'Rank {rank}/{numprocs} got beam data {len(beam_data)}')
        cas_filterbank.write(beam_data['cas'])
        ics_filterbank.write(beam_data['ics'])
        #vis.write(beam_data['vis'])


def dump_rankfile(values, fpga_per_rx=3):
    from craco import mpiutil
    hosts = mpiutil.parse_hostfile(values.hostfile)
    log.debug("Hosts %s", hosts)
    nrx = len(values.block)*len(values.card)*len(values.fpga)
    nbeams = values.nbeams
    nranks = nrx + nbeams
    total_cards = len(values.block)*len(values.card)
    ncards_per_host = (total_cards + len(hosts))//len(hosts)
    nrx_per_host = ncards_per_host*6//fpga_per_rx
    nbeams_per_host = (nbeams + len(hosts))//len(hosts)
    log.info(f'Spreading {nranks} over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas and {nbeams} beams with {nbeams_per_host} per host')

    rank = 0
    with open(values.dump_rankfile, 'w') as fout:
        # add all the beam processes
        for beam in range(nbeams):
            hostidx = rank // nbeams_per_host
            hostrank = rank % nbeams_per_host
            host = hosts[hostidx]
            slot = 0 # put on the U280 slot. If you put in slot1 it runs about 20% 
            core='5-6'
            s = f'rank {rank}={host} slot={slot}:{core} # Beam {beam}\n'
            fout.write(s)
            rank += 1

        rxrank = 0
        cardno = 0
        for block in values.block:
            for card in values.card:
                cardno += 1
                if values.max_ncards != 0 and cardno >= values.max_ncards + 1:
                    break
                for fpga in values.fpga[::fpga_per_rx]:
                    hostidx = rxrank // nrx_per_host
                    hostrank = rxrank % nrx_per_host
                    host = hosts[hostidx]
                    slot = 1 # fixed because both cards are on NUMA=1
                    # Put different FPGAs on differnt cores
                    evenfpga = fpga % 2 == 0
                    core = rxrank % 10
                    slot = 1
                    s = f'rank {rank}={host} slot={slot}:{core} # Block {block} card {card} fpga {fpga}\n'
                    fout.write(s)
                    rank += 1
                    rxrank += 1


class DummyVisSource:
    def __init__(self, values):
        self.nchans = values.nrx*values.nfpga_per_rx*NCHAN
        self.nt = values.nt
        self.npol = 1
        self.ra = 255.00
        self.dec = -30.00
        self.tstart = 888.88
        self.tsamp = 1.7e-3
        self.fch1 = 743.5
        self.foff = 0.166
        # beams first, then rx
    
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    cardcap.add_arguments(parser)
    parser.add_argument('--nbeams', type=int, help='Number of beams', default=36)
    parser.add_argument('--nfpga-per-rx', type=int, default=6, help='Number of FPGAS received by a single RX process')
    parser.add_argument('--vis-fscrunch', type=int, default=6, help='Amount to frequency average visibilities before transpose')
    parser.add_argument('--vis-tscrunch', type=int, default=1, help='Amount to time average visibilities before transpose')
    parser.add_argument('--cardcap-dir', '-D', help='Local directory (per node?) to load cardcap files from, if relevant. If unspecified, just use files from the positional arguments')
    parser.add_argument('--outdir', '-O', help='Directory to write outputs to', default='.')
    parser.add_argument(dest='files', nargs='*')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    if values.dump_rankfile:
        dump_rankfile(values, values.nfpga_per_rx)
        sys.exit(0)

    if values.max_ncards <= 0:
        ncards = len(values.block)*len(values.card)
    else:
        ncards = values.max_ncards

    values.nrx = ncards*len(values.fpga)//values.nfpga_per_rx
    values.nt = 64
    
    vis_source = DummyVisSource(values)

    if rank < values.nbeams:
        proc_beam(rank, vis_source, values)
    else:
        proc_rx(rank - values.nbeams, vis_source, values)

    comm.Barrier()
    print(f'Rank {rank}/{numprocs} complete')

                  
    

if __name__ == '__main__':
    _main()
