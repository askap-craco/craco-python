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
#import coloredlogs
import logging
#from mpi4py import MPI
import mpi4py
import time
import random

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

from rdma_transport import RdmaTransport
from rdma_transport import runMode
from rdma_transport import logType
from rdma_transport import ibv_wc

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2
log = logging.getLogger(__name__)

NFPGA_PER_LINK = 3

requestLogLevel = logType.LOG_DEBUG
messageSize = 65536
numMemoryBlocks = 1
numContiguousMessages = 1
dataFileName = None
numTotalMessages = 0
messageDelayTime = 0
identifierFileName = None 
metricURL = None
numMetricAveraging = 0

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
    transmitter_ids = np.arange(values.nlink)
    receive_ids = transmitter_ids % values.nrx
    my_transmitters = np.where(receive_ids == rank)[0]
    num_transmitters_sending_to_me = len(my_transmitters)

    log.info(f'Rank {rank} receiver expects data from {num_transmitters_sending_to_me} transmitters: {my_transmitters}')

    # Setup rdma receiver
    mode = runMode.RECV_MODE
    rdmaDeviceName = None #"mlx5_1"
    rdmaPort = 1
    gidIndex = -1
    
    rdma_receiver = RdmaTransport(requestLogLevel, 
                                  mode, 
                                  messageSize,
                                  numMemoryBlocks,
                                  numContiguousMessages,
                                  dataFileName,
                                  numTotalMessages,
                                  messageDelayTime,
                                  rdmaDeviceName,
                                  rdmaPort,
                                  gidIndex,
                                  identifierFileName,
                                  metricURL,
                                  numMetricAveraging)
    
    receiver_psn = rdma_receiver.getPacketSequenceNumber()
    receiver_qpn = rdma_receiver.getQueuePairNumber()
    receiver_gid = np.frombuffer(rdma_receiver.getGidAddress(), dtype=np.uint8)
    receiver_lid = rdma_receiver.getLocalIdentifier()
    receiver_info = {'rank':rank, 'psn':receiver_psn, 'qpn': receiver_qpn,
                     'gid': receiver_gid, 'lid':receiver_lid}

    # Send info to my transmitters
    status = MPI.Status()
    for tx in my_transmitters:
        transmitter_rank = tx + values.nrx
        log.info(f'Sending the receiver info {receiver_info} to a transmitter with rank {transmitter_rank}')
        world.send(receiver_info, dest=int(transmitter_rank), tag=1)

    # recv informaton from transmitters
    for tx in my_transmitters:
        transmitter_rank = tx + values.nrx
        transmitter_info = world.recv(source=transmitter_rank, tag=MPI.ANY_TAG, status=status)
        log.info(f'Got transmitter info {transmitter_info} from a transmitter with rank={status.Get_source()} tag={status.Get_tag()}')

        transmitter_psn = transmitter_info['psn']
        transmitter_qpn = transmitter_info['qpn']
        transmitter_gid = transmitter_info['gid']
        transmitter_lid = transmitter_info['lid']

    msg = np.zeros(values.msg_size)
    start = time.time()
    for imsg in range(values.nmsg):
        # TODO: extra loop over the number of transmitters I'm exxpecting
        for imsg2 in range(num_transmitters_sending_to_me):
            if values.method == 'mpi':
                world.Recv(msg, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                log.debug(f'Receviver with {rank} got data from transmitter={status.Get_source()} tag={status.Get_tag()} mean={msg.mean()}')

    end = time.time()
    interval = end - start
    rate = msg.itemsize*msg.size*values.nmsg*num_transmitters_sending_to_me*8/float(interval)/1e9
    log.info(f'Rank {rank} receiver received data at {rate} Gbps')
        

def be_transmitter(values):
    assert values.nlink >= values.nrx, 'Each transmitter only sends to one place'
    transmitters = world.Split(0, rank)
    assert transmitters.Get_size() == values.nlink
    
    receivers = world.Split(1, rank)

    # the rank of us in the list of transmitters
    transmitter_rank = transmitters.Get_rank()
    assert transmitter_rank == world.Get_rank() - values.nrx
    assert transmitter_rank < values.nlink

    # the reciver rank we'll send our data  to
    receiver_rank = transmitter_rank % values.nrx

    # Wait for info from my receiver
    status = MPI.Status()
    receiver_info = world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    log.info(f'Got receiver info {receiver_info} from receiver with rank={status.Get_source()} tag={status.Get_tag()}')

    receiver_psn = receiver_info['psn']
    receiver_qpn = receiver_info['qpn']
    receiver_gid = receiver_info['gid']
    receiver_lid = receiver_info['lid']

    # Setup rdma transmitter 
    mode = runMode.SEND_MODE
    rdmaDeviceName = None #"mlx5_1"
    rdmaPort = 1
    gidIndex = -1
    
    rdma_transmitter = RdmaTransport(requestLogLevel, 
                                  mode, 
                                  messageSize,
                                  numMemoryBlocks,
                                  numContiguousMessages,
                                  dataFileName,
                                  numTotalMessages,
                                  messageDelayTime,
                                  rdmaDeviceName,
                                  rdmaPort,
                                  gidIndex,
                                  identifierFileName,
                                  metricURL,
                                  numMetricAveraging)
    
    transmitter_psn = rdma_transmitter.getPacketSequenceNumber()
    transmitter_qpn = rdma_transmitter.getQueuePairNumber()
    transmitter_gid = np.frombuffer(rdma_transmitter.getGidAddress(), dtype=np.uint8)
    transmitter_lid = rdma_transmitter.getLocalIdentifier()
    transmitter_info = {'rank':rank, 'psn':transmitter_psn, 'qpn': transmitter_qpn,
                        'gid': transmitter_gid, 'lid':transmitter_lid}

    log.info(f'Sending transmitter info {transmitter_info} to a receiver with rank {receiver_rank}')
    world.send(transmitter_info, dest=int(receiver_rank), tag=1)
    
    msg = np.zeros(values.msg_size)
    for imsg in range(values.nmsg):
        log.debug(f'Sending msg {imsg} from transmitter {transmitter_rank} to receiver {receiver_rank}')
        if values.method == 'mpi':
            world.Send(msg+imsg, dest=receiver_rank, tag=transmitter_rank)
        
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

    #coloredlogs.install(level='DEBUG', logger=log)
    
    #if values.verbose:
    #    coloredlogs.install(
    #        fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
    #        level='DEBUG')
    #else:            
    #    coloredlogs.install(
    #        fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
    #        level='INFO')
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
