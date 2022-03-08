#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
import mpi4py
mpi4py.rc.threads = False
from mpi4py import MPI

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)
values = None
ncoarse_chan = 4
nt_per_bf_frame = 2048//16
nfine_chan = 6
nbl = 435
ncards = 1

# Can't use scatter / gather as you get the data back in your rank!


def be_rx(values):
    nbeam = values.nbeam
    card_data_shape = [nbeam, ncoarse_chan, nt_per_bf_frame, nfine_chan, nbl, 2]
    my_card_data = np.zeros((card_data_shape), dtype=np.int16)
    my_card_data.flat = np.arange(np.prod(card_data_shape))

    # I participate in nbeams communicators
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    log.info('I am %d a RX', rank)
    comm.Barrier()

    for b in range(nbeam):
        brank = values.nrx + b
        log.info('I am %d sending to %d', rank, brank)
        comm.Send(my_card_data[b,:], brank)
        

def be_beam(values):
    # I participate in only 1 of the communicators - but I need to call split nbeams times anyway
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    my_beam = rank - values.nrx
    nbeam = values.nbeam
    assert my_beam < nbeam
    card_data_shape = [values.nrx, ncoarse_chan, nt_per_bf_frame, nfine_chan, nbl, 2]
    my_card_data = np.zeros((card_data_shape), dtype=np.int16)
    log.info('I am rank %d i am beam %d', rank, my_beam)
    comm.Barrier()

    for ir in range(values.nrx):
        status = MPI.Status()
        comm.probe(status=status)
        comm.Recv(my_card_data[ir, :])

        log.info('Got data %s from count=%s source=%s', my_card_data.shape, status.Get_count(), status.Get_source())
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--nbeam', type=int)
    parser.add_argument('--nrx', type=int)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    assert numprocs == values.nbeam + values.nrx

    if rank < values.nrx:
        be_rx(values)
    else:
        be_beam(values)


    
    

if __name__ == '__main__':
    _main()
