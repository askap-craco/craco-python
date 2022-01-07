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
import mpi4py
import time
import random

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

from rdma_transport import RdmaTransport
from rdma_transport import runMode
from rdma_transport import ibv_wc

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 10000
# mpirun -c 2 run_cluster_messages.py --nrx 1 --nlink 1 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 10000

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method mpi --msg-size 65536 --nmsg 10000
# mpirun -c 2 run_cluster_messages.py --nrx 1 --nlink 1 --method mpi --msg-size 65536 --nmsg 10000

# cpu binding

# numCompltions has numMissing included??? check RDMAapi.c to figure out

log = logging.getLogger(__name__)

NFPGA_PER_LINK = 3

# we do not need these numbers configurable as barrier will sovle any sync issue
messageDelayTimeRecv = 0
messageDelayTimeSend = 0

# We do not use identifier file here, so we hard code it
identifierFileName = None 
            
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

def receive_with_mpi(values, status, num_transmitters):
    # Receive messages
    msg = np.zeros(values.msg_size)
    
    start = time.time()
    for imsg in range(values.nmsg):
        # TODO: extra loop over the number of transmitters I'm exxpecting
        for tx in range(num_transmitters):
            world.Recv(msg, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            log.debug(f'Receviver with {rank} got data from transmitter={status.Get_source()} tag={status.Get_tag()} mean={msg.mean()}')
    end = time.time()
    
    interval = end - start
    rate = msg.itemsize*msg.size*values.nmsg*num_transmitters*8.E-9/float(interval)
    log.info(f'Rank {rank} receiver elapsed time is {interval} seconds')
    log.info(f'Rank {rank} receiver data rate is {rate} Gbps')

def transmit_with_mpi(values, status, receiver_rank, transmitter_rank):
    # Get messages now
    msg = np.zeros(values.msg_size)
    
    start = time.time()
    for imsg in range(values.nmsg):
        log.debug(f'Sending msg {imsg} from transmitter {transmitter_rank} to receiver {receiver_rank}')
        world.Send(msg+imsg, dest=receiver_rank, tag=transmitter_rank)
        
    end = time.time()
    interval = end - start
    rate = msg.itemsize*msg.size*values.nmsg*8.E-9/float(interval)
    log.info(f'Rank {transmitter_rank} transmitter elapsed time is {interval} seconds')
    log.info(f'Rank {transmitter_rank} transmitter data rate is {rate} Gbps')

def create_rdma_receivers(values, my_transmitters):
    # Setup rdma receiver
    mode = runMode.RECV_MODE
    rdmaDeviceName = None #"mlx5_1"
    rdmaPort = 1
    gidIndex = -1
    
    # Send info to my transmitters
    rdma_receivers = []
    for tx in my_transmitters:
        rdma_receiver = RdmaTransport(mode, 
                                      values.msg_size,
                                      values.num_blks,
                                      values.num_cmsgs,
                                      values.nmsg,
                                      messageDelayTimeRecv,
                                      rdmaDeviceName,
                                      rdmaPort,
                                      gidIndex)
        rdma_receivers.append(rdma_receiver)

    return rdma_receivers

def create_rdma_transmitter(values):
    # Setup rdma transmitter 
    mode = runMode.SEND_MODE
    rdmaDeviceName = None #"mlx5_1"
    rdmaPort = 1
    gidIndex = -1
    
    rdma_transmitter = RdmaTransport(mode, 
                                     values.msg_size,
                                     values.num_blks,
                                     values.num_cmsgs,
                                     values.nmsg,
                                     messageDelayTimeSend,
                                     rdmaDeviceName,
                                     rdmaPort,
                                     gidIndex)

    return rdma_transmitter

def send_receivers_info(values, rdma_receivers, my_transmitters):
    
    for tx in my_transmitters:            
        rdma_receiver_psn = rdma_receivers[tx].getPacketSequenceNumber()
        rdma_receiver_qpn = rdma_receivers[tx].getQueuePairNumber()
        rdma_receiver_gid = np.frombuffer(rdma_receivers[tx].getGidAddress(), dtype=np.uint8)
        rdma_receiver_lid = rdma_receivers[tx].getLocalIdentifier()
        rdma_receiver_info = {'rank':rank, 'psn':rdma_receiver_psn, 'qpn': rdma_receiver_qpn,
                              'gid': rdma_receiver_gid, 'lid':rdma_receiver_lid}
        
        transmitter_rank = tx + values.nrx
        log.info(f'Sending the rdma receiver info {rdma_receiver_info} to a rdma transmitter with rank {transmitter_rank}')
        world.send(rdma_receiver_info, dest=int(transmitter_rank), tag=1)

def send_transmitter_info(rdma_transmitter, receiver_rank):
    rdma_transmitter_psn = rdma_transmitter.getPacketSequenceNumber()
    rdma_transmitter_qpn = rdma_transmitter.getQueuePairNumber()
    rdma_transmitter_gid = np.frombuffer(rdma_transmitter.getGidAddress(), dtype=np.uint8)
    rdma_transmitter_lid = rdma_transmitter.getLocalIdentifier()

    rdma_transmitter_info = {'rank':rank, 'psn':rdma_transmitter_psn, 'qpn': rdma_transmitter_qpn,
                             'gid': rdma_transmitter_gid, 'lid':rdma_transmitter_lid}
    
    log.info(f'Sending rdma transmitter info {rdma_transmitter_info} to a rdma receiver with rank {receiver_rank}')
    world.send(rdma_transmitter_info, dest=int(receiver_rank), tag=1)
        
def pair_with_transmitters(values, rdma_receivers, my_transmitters, status):
    # recv informaton from transmitters
    for tx in my_transmitters:
        transmitter_rank = tx + values.nrx
        
        rdma_transmitter_info = world.recv(source=transmitter_rank, tag=MPI.ANY_TAG, status=status)
        log.info(f'Got transmitter info {rdma_transmitter_info} from a rdma transmitter with rank={status.Get_source()} tag={status.Get_tag()}')
        
        rdma_transmitter_psn = rdma_transmitter_info['psn']
        rdma_transmitter_qpn = rdma_transmitter_info['qpn']
        rdma_transmitter_gid = rdma_transmitter_info['gid']
        rdma_transmitter_lid = rdma_transmitter_info['lid']
        
        rdma_receivers[tx].setPacketSequenceNumber(rdma_transmitter_psn)
        rdma_receivers[tx].setQueuePairNumber(rdma_transmitter_qpn)
        rdma_receivers[tx].setGidAddress(rdma_transmitter_gid)
        rdma_receivers[tx].setLocalIdentifier(rdma_transmitter_lid)
        rdma_receivers[tx].setupRdma(identifierFileName)

def pair_with_receiver(rdma_transmitter, identifierFileName, status):
    rdma_receiver_info = world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    log.info(f'Got rdma receiver info {rdma_receiver_info} from rdma receiver with rank={status.Get_source()} tag={status.Get_tag()}')
    
    rdma_receiver_psn = rdma_receiver_info['psn']
    rdma_receiver_qpn = rdma_receiver_info['qpn']
    rdma_receiver_gid = rdma_receiver_info['gid']
    rdma_receiver_lid = rdma_receiver_info['lid']

    # now setup remote infromation and finish rdma setup
    rdma_receiver_psn = rdma_receiver_info['psn']
    rdma_receiver_qpn = rdma_receiver_info['qpn']
    rdma_receiver_gid = rdma_receiver_info['gid']
    rdma_receiver_lid = rdma_receiver_info['lid']
    
    rdma_transmitter.setPacketSequenceNumber(rdma_receiver_psn)
    rdma_transmitter.setQueuePairNumber(rdma_receiver_qpn)
    rdma_transmitter.setGidAddress(rdma_receiver_gid)
    rdma_transmitter.setLocalIdentifier(rdma_receiver_lid)
    
    rdma_transmitter.setupRdma(identifierFileName)

def setup_buffers_for_single_rdma(values, rdma):
    rdma_buffers = []
    for iblock in range(values.num_blks):
        rdma_memory = rdma.get_memoryview(iblock)

        # for now use int8 as it is the same size as char
        rdma_buffer = np.frombuffer(rdma_memory, dtype=np.int8)

        rdma_buffer[:] = 0
        rdma_buffers.append(rdma_buffer)
        
    return rdma_buffers
    
def setup_buffers_for_multiple_rdma(values, rdma_receivers, my_transmitters):
    rdma_buffers = []
    for tx in my_transmitters:
        rdma_buffers.append(setup_buffers_for_single_rdma(values, rdma_receivers[tx]))

    return rdma_buffers

def be_receiver(values):
    receivers = world.Split(1, rank)
    transmitters = world.Split(0, rank)
    transmitter_ids = np.arange(values.nlink)
    receive_ids = transmitter_ids % values.nrx
    my_transmitters = np.where(receive_ids == rank)[0]
    num_transmitters = len(my_transmitters)

    log.info(f'Rank {rank} receiver expects data from {num_transmitters} transmitters: {my_transmitters}')

    status = MPI.Status()

    if values.method == 'mpi':
        receive_with_mpi(values, status, num_transmitters)
        
    if values.method == 'rdma':
        rdma_receivers = create_rdma_receivers(values, my_transmitters)
        send_receivers_info(values, rdma_receivers, my_transmitters)
        pair_with_transmitters(values, rdma_receivers, my_transmitters, status)
        rdma_buffers = setup_buffers_for_multiple_rdma(values, rdma_receivers, my_transmitters)
        
        print(f'rdma_buffers for receiver shape is {np.array(rdma_buffers).shape}')
        start = time.time()
        
        numMissingTotal = np.zeros(num_transmitters, dtype=int)
        numMessagesTotal = np.zeros(num_transmitters, dtype=int)
        numCompletionsTotal = np.zeros(num_transmitters, dtype=int)
        tx = 0

        while numMessagesTotal[tx] < values.nmsg:
            rdma_receivers[tx].issueRequests()
            world.Barrier()
            
            rdma_receivers[tx].waitRequestsCompletion()
            rdma_receivers[tx].pollRequests()

            numCompletionsFound = rdma_receivers[tx].get_numCompletionsFound()
            numMissingFound     = rdma_receivers[tx].get_numMissingFound()

            numCompletionsTotal[tx] += numCompletionsFound
            numMissingTotal[tx]     += numMissingFound
            numMessagesTotal[tx]    += (numCompletionsFound+numMissingFound)

            workCompletions = rdma_receivers[tx].get_workCompletions()

            if values.test == 'throughput':
                continue
            else:
                for i in range(numCompletionsFound):
                    index = workCompletions[i].wr_id
                    
                    # Get data for buffer regions
                    block_index = index//values.num_cmsgs
                    
                    # now it is data for each message
                    message_index = index%values.num_cmsgs

                    sum_data = np.sum(rdma_buffers[tx][block_index][0:10])
                    if values.test == 'ones':
                        assert sum_data == 10
                    if values.test == 'increment':
                        #print(block_index, sum_data)
                        assert sum_data == 10*block_index
                        
                        
        end = time.time()
        interval = end - start
        rate = values.msg_size*numCompletionsTotal*num_transmitters*8.E-9/float(interval)
        
        log.info(f'Rank {rank} receiver elapsed time is {interval} seconds')
        log.info(f'Rank {rank} receiver data rate is {rate} Gbps')
        log.info(f'Rank {rank} receiver, message missed is {numMissingTotal[tx]}')
        log.info(f'Rank {rank} receiver, message received is {numCompletionsTotal[tx]}')
        log.info(f'Rank {rank} receiver, message total is {values.nmsg}')
        log.info(f'Rank {rank} receiver, message loss rate is {numMissingTotal[tx]/float(numMessagesTotal[tx])}')
        
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

    if values.method == 'mpi':
        transmit_with_mpi(values, status, receiver_rank, transmitter_rank)
        
    if values.method == 'rdma':
        # Setup rdma transmitter 
        rdma_transmitter = create_rdma_transmitter(values)
        send_transmitter_info(rdma_transmitter, receiver_rank)
        pair_with_receiver(rdma_transmitter, identifierFileName, status)

        rdma_buffers = setup_buffers_for_single_rdma(values, rdma_transmitter)

        print(f'rdma_buffers for transmitter shape is {np.array(rdma_buffers).shape}')

        if values.test == 'ones':
            for i in range(values.num_blks):
                rdma_buffers[i][0:10] = 1
                
        if values.test == 'increment':
            for i in range(values.num_blks):
                rdma_buffers[i][0:10] = i

        start = time.time()
        numCompletionsTotal = 0
        while numCompletionsTotal < values.nmsg:
            world.Barrier()
            rdma_transmitter.issueRequests()
            rdma_transmitter.waitRequestsCompletion()
            rdma_transmitter.pollRequests()
            numCompletionsFound = rdma_transmitter.get_numCompletionsFound()
            numCompletionsTotal += numCompletionsFound
            
            #workCompletions = rdma_transmitter.get_workCompletions()
       
        end = time.time()
        interval = end - start

        rate = values.msg_size*values.nmsg*8.E-9/float(interval)
        log.info(f'Rank {transmitter_rank} transmitter elapsed time is {interval} seconds')
        log.info(f'Rank {transmitter_rank} transmitter data rate is {rate} Gbps')
        
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nrx', type=int, help='Number of receivers', default=1)
    parser.add_argument('--nlink', type=int, help='Number of transmit links (2x number of cards)', default=1)
    parser.add_argument('--nmsg', type=int, default=1000)
    parser.add_argument('--msg-size', type=int, default=65536)
    parser.add_argument('--num-blks', type=int, default=10)
    parser.add_argument('--num-cmsgs', type=int, default=100)
    parser.add_argument('--method', default='mpi', help='mpi or rdma')
    parser.add_argument('--test', default='throughput', help='throughput, ones or increment')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    assert size == values.nrx + values.nlink

    if values.method == 'mpi':
        assert values.nmsg != 0
    if values.method == 'rdma':
        assert values.num_blks != 0
        assert values.num_cmsgs != 0
        
        if values.nmsg == 0:
            values.nmsg = values.num_blks*values.num_cmsgs

    if rank < values.nrx:
        be_receiver(values)
    else:
        be_transmitter(values)

if __name__ == '__main__':
    _main()
