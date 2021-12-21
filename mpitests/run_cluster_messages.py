#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2020
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from mpi4py import MPI
import mpi4py
import time
import random

log = logging.getLogger(__name__)

NFPGA_PER_LINK = 3

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

'''
Each card has 4 MHz of bandwidth = 6 FPGAS and 36 beams, and produces datda in blcoks of 128 time samples
Each link has half the bandwdith / channels from 3 FPGAS. 
Each 1 MHz has 6 channels.
Tint = 1.7 ms or so.

Each link will send data to only one receiver because nrx < ntx

Over 6 blocks x 12 cards each x = 72 cards = 144 links = 288 MHz = 1728 channels 

We share the data amongst NRX  = 20 receivers by channel

Later on we need to tranpsose with MPI

'''

world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

def be_receiver(values):
    receivers = world.Split(1, rank)
    transmitters = world.Split(0, rank)
    transmit_ids = np.arange(values.nlink)
    receive_ids = transmit_ids % values.nrx
    my_transmitters = np.where(receive_ids == rank)[0]
    num_transmitters_sending_to_me = len(my_transmitters)
    log.info(f'Rank {rank} expects data from {num_transmitters_sending_to_me} transmitters:{my_transmitters}')
    receiver_info = {'rank':rank, 'psn':random.randint(0, 16384), 'qpn': random.randint(0, 16384),
                     'gid': np.random.bytes(16), 'lid':np.random.randint(0,16384)}
    

    # Send info to my transmitters
    for tx in my_transmitters:
        destrank = tx + values.nrx
        log.info(f'Sending {receiver_info} to rank {destrank}')
        world.send(receiver_info, dest=int(destrank), tag=1)

    msg = np.zeros(values.msg_size)
    status = MPI.Status()
    start = time.time()
    for imsg in range(values.nmsg):
        # TODO: extra loop over the number of transmitters I'm exxpecting
        for imsg2 in range(num_transmitters_sending_to_me):
            if values.method == 'mpi':
                world.Recv(msg, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                log.debug(f'Receviver {rank} Got data from transmitter={status.Get_source()} tag={status.Get_tag()} mean={msg.mean()}')

    end = time.time()
    interval = end - start
    rate = msg.itemsize*msg.size*values.nmsg*num_transmitters_sending_to_me*8/float(interval)/1e9
    log.info(f'Rank {rank} Received data at {rate} Gbps')
        

def be_transmitter(values):
    assert values.nlink >= values.nrx, 'Each transmitter only sends to one place'
    transmitters = world.Split(0, rank)
    assert transmitters.Get_size() == values.nlink
    
    receivers = world.Split(1, rank)

    # the rank of us in the list of transmitters
    transmit_rank = transmitters.Get_rank()
    assert transmit_rank == world.Get_rank() - values.nrx
    assert transmit_rank < values.nlink

    # the reciver rank we'll send our data  to
    target_receiver_rank = transmit_rank % values.nrx

    # Wait for info from my receiver
    status = MPI.Status()
    info = world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    log.info(f'Got {info} from rank={status.Get_source()} tag={status.Get_tag()}')
    
    msg = np.zeros(values.msg_size)
    for imsg in range(values.nmsg):
        log.debug(f'Sending msg {imsg} from {transmit_rank} to {target_receiver_rank}')
        if values.method == 'mpi':
            world.Send(msg+imsg, dest=target_receiver_rank, tag=transmit_rank)
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nrx', type=int, help='Number of receivers', default=1)
    parser.add_argument('--nlink', type=int, help='Number of transmit links (2x number of cards)', default=1)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--nmsg', type=int, default=1000)
    parser.add_argument('--msg-size', type=int, default=4096)
    parser.add_argument('--method', default='mpi', help='mpi or rdma')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    assert size == values.nrx + values.nlink

    # ranks < nrx are receivers
    # ransk >= nrx are transmitter
    log.info(f"World rank {rank} size {size}")

    if rank < values.nrx:
        be_receiver(values)
    else:
        be_transmitter(values)


    
    

if __name__ == '__main__':
    _main()
