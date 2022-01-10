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
from rdma_transport import ibv_wc_status

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 10_000 --test ones --send-delay 10_000
# mpirun -c 2 run_cluster_messages.py --nrx 1 --nlink 1 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 10_000 --test ones --send-delay 10_000

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method mpi --msg-size 65_536 --nmsg 10_000
# mpirun -c 2 run_cluster_messages.py --nrx 1 --nlink 1 --method mpi --msg-size 65_536 --nmsg 10_000

# cpu binding
# if there is message missing, the result will be wrong, add send-delay option so that we can make message missing free for result check
# throughput test does not need delay option

log = logging.getLogger(__name__)

NFPGA_PER_LINK = 3

# we do not need these numbers configurable as barrier will sovle any sync issue
messageDelayTimeRecv = 0

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

def create_rdma_receivers(values, num_transmitters):
    # Setup rdma receiver
    mode = runMode.RECV_MODE
    rdmaDeviceName = None #"mlx5_1"
    rdmaPort = 1
    gidIndex = -1
    
    # Send info to my transmitters
    rdma_receivers = []
    for tx in range(num_transmitters):
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
                                     values.send_delay,
                                     rdmaDeviceName,
                                     rdmaPort,
                                     gidIndex)

    return rdma_transmitter

def send_receivers_info(values, rdma_receivers, num_transmitters):
    
    for tx in range(num_transmitters):            
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
        
def pair_with_transmitters(values, rdma_receivers, num_transmitters, status):
    # recv informaton from transmitters
    for tx in range(num_transmitters):
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

        # for now use int8 as it is the same size as char
        shape = (values.num_cmsgs, values.msg_size)
        rdma_buffers.append(rdma_buffer.reshape(shape))
        
    return rdma_buffers
    
def setup_buffers_for_multiple_rdma(values, rdma_receivers, num_transmitters):
    rdma_buffers = []
    for tx in range(num_transmitters):
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
        rdma_receivers = create_rdma_receivers(values, num_transmitters)
        send_receivers_info(values, rdma_receivers, num_transmitters)
        pair_with_transmitters(values, rdma_receivers, num_transmitters, status)
        rdma_buffers = setup_buffers_for_multiple_rdma(values, rdma_receivers, num_transmitters)
        
        log.info(f'rdma_buffers for receiver shape is {np.array(rdma_buffers).shape}')
        
        numMissingTotal = np.zeros(num_transmitters, dtype=int)
        numMessagesTotal = np.zeros(num_transmitters, dtype=int)
        numCompletionsTotal = np.zeros(num_transmitters, dtype=int)
        
        for tx in range(num_transmitters):
            rdma_receivers[tx].issueRequests()

        # setup while loop stop marker
        do_while = False
        for tx in range(num_transmitters):
            do_while = do_while or (numMessagesTotal[tx] < values.nmsg)
            
        world.Barrier() # receiver should be ready before transmitter is ready
        start = time.time()
        while do_while:
            for tx in range(num_transmitters):
                if numMessagesTotal[tx] < values.nmsg:
                    rdma_receivers[tx].waitRequestsCompletion()
                
            for tx in range(num_transmitters):
                if numMessagesTotal[tx] < values.nmsg:
                    rdma_receivers[tx].pollRequests()

            for tx in range(num_transmitters):
                if numMessagesTotal[tx] < values.nmsg:
                    numCompletionsFound = rdma_receivers[tx].get_numCompletionsFound()
                    numMissingFound     = rdma_receivers[tx].get_numMissingFound()

                    numCompletionsTotal[tx] += numCompletionsFound
                    numMissingTotal[tx]     += numMissingFound
                    numMessagesTotal[tx]    += (numCompletionsFound+numMissingFound)

                    # we do not need following with throughput test
                    if values.test is not 'throughput':
                        workCompletions = rdma_receivers[tx].get_workCompletions()
                        assert numCompletionsFound == len(workCompletions)
                        
                        #for i in range(numCompletionsFound):
                        for workCompletion in workCompletions:
                            #workCompletion = workCompletions[i]
                            if workCompletion.status == ibv_wc_status.IBV_WC_SUCCESS:
                                index = workCompletion.wr_id
                        
                                # Get data for buffer regions
                                block_index = index//values.num_cmsgs
                    
                                # now it is data for each message
                                message_index = index%values.num_cmsgs
                        
                                sum_data = np.sum(rdma_buffers[tx][block_index][message_index,0:10])
                                if values.test == 'ones':
                                    #log.info(sum_data)
                                    assert sum_data == 10, f'Invalid sum_data {sum_data}'
                                if values.test == 'increment':
                                    #log.info(f'{sum_data}, {block_index}, {rdma_buffers[tx][block_index][message_index,0:10]}')
                                    assert sum_data == 10*block_index, f'Invalid sum_data {sum_data} at {block_index}'
                        
            for tx in range(num_transmitters):
                if numMessagesTotal[tx] < values.nmsg:
                    rdma_receivers[tx].issueRequests()

            # check if we need to stop while loop
            do_while = False
            for tx in range(num_transmitters):
                do_while = do_while or (numMessagesTotal[tx] < values.nmsg)

        # we may need to get end and interval for all transmitter seperately
        end = time.time()
        interval = end - start

        for tx in range(num_transmitters):
            rate = values.msg_size*numCompletionsTotal[tx]*8.E-9/float(interval)
            
            log.info(f'Rank {rank} receiver from transmitter {tx}, elapsed time is {interval} seconds')
            log.info(f'Rank {rank} receiver from transmitter {tx}, data rate is {rate} Gbps')
            
            log.info(f'Rank {rank} receiver from transmitter {tx}, message missed is {numMissingTotal[tx]}')
            log.info(f'Rank {rank} receiver from transmitter {tx}, message received is {numCompletionsTotal[tx]}')
            log.info(f'Rank {rank} receiver from transmitter {tx}, message total is {values.nmsg}')
            log.info(f'Rank {rank} receiver from transmitter {tx}, message loss rate is {numMissingTotal[tx]/float(numMessagesTotal[tx])}\n')
            
def be_transmitter(values):
    assert values.nlink >= values.nrx, 'Each transmitter only sends to one place'
    transmitters = world.Split(0, rank)
    #assert transmitters.Get_size() == values.nlink, f'{transmitters.Get_size()} {values.nlink}'
    
    receivers = world.Split(1, rank)

    # the rank of us in the list of transmitters
    transmitter_rank = transmitters.Get_rank()
    assert transmitter_rank == world.Get_rank() - values.nrx
    assert transmitter_rank < values.nlink, f'{transmitter_rank} {values.nlink}'

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

        log.info(f'rdma_buffers for transmitter shape is {np.array(rdma_buffers).shape}')

        if values.test == 'ones':
            for i in range(values.num_blks):
                for j in range(values.num_cmsgs):
                    rdma_buffers[i][j,0:10] = 1
                
        if values.test == 'increment':
            for i in range(values.num_blks):
                for j in range(values.num_cmsgs):
                    rdma_buffers[i][j,0:10] = i
                    #assert sum(rdma_buffers[i][j,0:10]) == 10*i
                    
        numCompletionsTotal = 0
        world.Barrier() # receiver should be ready before transmitter is ready
        start = time.time()
        while numCompletionsTotal < values.nmsg:
            rdma_transmitter.issueRequests()
            rdma_transmitter.waitRequestsCompletion()
            rdma_transmitter.pollRequests()
            numCompletionsFound = rdma_transmitter.get_numCompletionsFound()
            numCompletionsTotal += numCompletionsFound
            
            workCompletions = rdma_transmitter.get_workCompletions()
       
        end = time.time()
        interval = end - start

        rate = values.msg_size*values.nmsg*8.E-9/float(interval)
        log.info(f'Rank {transmitter_rank} transmitter elapsed time is {interval} seconds')
        log.info(f'Rank {transmitter_rank} transmitter data rate is {rate} Gbps\n')
        
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nrx', type=int, help='Number of receivers', default=1)
    parser.add_argument('--nlink', type=int, help='Number of transmit links (2x number of cards)', default=1)
    parser.add_argument('--nmsg', type=int, default=1000)
    parser.add_argument('--msg-size', type=int, default=65536)
    parser.add_argument('--num-blks', type=int, default=10)
    parser.add_argument('--num-cmsgs', type=int, default=100)
    parser.add_argument('--num_node', type=int, default=2)
    parser.add_argument('--send-delay', type=int, help='delay in microseconds for RDMA sender/transmitter', default=0)
    parser.add_argument('--method', default='mpi', help='mpi or rdma')
    parser.add_argument('--test', default='throughput', help='throughput, ones or increment')
        
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    #assert size == values.num_node*(values.nrx + values.nlink), f'{size} {values.nrx} {values.nlink}'
    assert size == values.nrx + values.nlink, f'{size} {values.nrx} {values.nlink}'

    if values.method == 'mpi':
        assert values.nmsg != 0
    if values.method == 'rdma':
        assert values.num_blks != 0
        assert values.num_cmsgs != 0
        assert values.send_delay >= 0
        
        if values.nmsg == 0:
            values.nmsg = values.num_blks*values.num_cmsgs

    if rank < values.nrx:
        be_receiver(values)
    else:
        be_transmitter(values)

if __name__ == '__main__':
    _main()
