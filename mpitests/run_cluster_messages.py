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
from rdma_transport import logType
from rdma_transport import ibv_wc

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2
# receiver will hang if there are missed messages

log = logging.getLogger(__name__)

NFPGA_PER_LINK = 3

#requestLogLevel = logType.LOG_DEBUG
requestLogLevel = logType.LOG_INFO
messageSize = 65536
numMemoryBlocks = 10
numContiguousMessages = 100
numRepeat = 200
dataFileName = None
numTotalMessages = numRepeat*numMemoryBlocks*numContiguousMessages
messageDelayTimeRecv = 0
messageDelayTimeSend = 0#1000
identifierFileName = None 
metricURL = None
numMetricAveraging = 0

ndataPrint = 10
            
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

    status = MPI.Status()

    # Receive messages
    msg = np.zeros(values.msg_size)
    
    if values.method == 'mpi':
        for imsg in range(values.nmsg):
            # TODO: extra loop over the number of transmitters I'm exxpecting
            for tx in range(num_transmitters_sending_to_me):
                world.Recv(msg, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                log.debug(f'Receviver with {rank} got data from transmitter={status.Get_source()} tag={status.Get_tag()} mean={msg.mean()}')

    if values.method == 'rdma':
        # Setup rdma receiver
        mode = runMode.RECV_MODE
        rdmaDeviceName = None #"mlx5_1"
        rdmaPort = 1
        gidIndex = -1
        
        # Send info to my transmitters
        rdma_receivers = []
        for tx in my_transmitters:
            rdma_receiver = RdmaTransport(requestLogLevel, 
                                          mode, 
                                          messageSize,
                                          numMemoryBlocks,
                                          numContiguousMessages,
                                          dataFileName,
                                          numTotalMessages,
                                          messageDelayTimeRecv,
                                          rdmaDeviceName,
                                          rdmaPort,
                                          gidIndex,
                                          metricURL,
                                          numMetricAveraging)
            rdma_receivers.append(rdma_receiver)
            
            rdma_receiver_psn = rdma_receiver.getPacketSequenceNumber()
            rdma_receiver_qpn = rdma_receiver.getQueuePairNumber()
            rdma_receiver_gid = np.frombuffer(rdma_receiver.getGidAddress(), dtype=np.uint8)
            rdma_receiver_lid = rdma_receiver.getLocalIdentifier()
            rdma_receiver_info = {'rank':rank, 'psn':rdma_receiver_psn, 'qpn': rdma_receiver_qpn,
                                  'gid': rdma_receiver_gid, 'lid':rdma_receiver_lid}
            
            transmitter_rank = tx + values.nrx
            log.info(f'Sending the rdma receiver info {rdma_receiver_info} to a rdma transmitter with rank {transmitter_rank}')
            world.send(rdma_receiver_info, dest=int(transmitter_rank), tag=1)

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
            rdma_receivers[tx].setupRdma(identifierFileName);

        rdma_buffer = []
        for tx in my_transmitters:
            rdma_buffer_tx = []
            for iblock in range(numMemoryBlocks):
                rdma_memory = rdma_receivers[tx].get_memoryview(iblock)
                rdma_buffer_tx.append(np.frombuffer(rdma_memory, dtype=np.int16))
            rdma_buffer.append(rdma_buffer_tx)

        start = time.time()
        world.Barrier()
        
        numMissingTotal = 0
        numMessagesTotal = 0
        numCompletionsTotal = 0
        while numMessagesTotal < values.nmsg:
            print(f'Receiver {numCompletionsTotal} VS {values.nmsg}')
            rdma_receivers[tx].issueRequests()
            rdma_receivers[tx].waitRequestsCompletion()
            rdma_receivers[tx].pollRequests()

            numCompletionsFound = rdma_receivers[tx].get_numCompletionsFound()
            numMissingFound     = rdma_receivers[tx].get_numMissingFound()

            numCompletionsTotal += numCompletionsFound
            numMissingTotal     += numMissingFound
            numMessagesTotal    += (numCompletionsFound+numMissingFound)

            workCompletions = rdma_receivers[tx].get_workCompletions()
            
            for i in range(numCompletionsFound):
                index = workCompletions[i].wr_id

                # Get data for buffer regions
                block_index = index//numContiguousMessages
                #rdma_memory = rdma_receivers[tx].get_memoryview(block_index)
                #rdma_buffer = np.frombuffer(rdma_memory, dtype=np.int16) # 163 and 164 will be slow, do it at the begning

                # now it is data for each message
                message_index = index%numContiguousMessages
                data_start_index = int(message_index*messageSize*8//16) # because messageSize is in bytes, but we are using uint16_t, which is 2 bytes

                #print(f"receiver {rank} data index {i}")
                
                #print(f'The first {ndataPrint} data of rdma_receiver {rank} receivered from rdma transmitter {status.Get_source()} is {rdma_buffer[data_start_index:data_start_index+ndataPrint]}')
                
    end = time.time()
    interval = end - start
    #rate = msg.itemsize*msg.size*values.nmsg*num_transmitters_sending_to_me*8/float(interval)/1e9
    #required_rate = msg.itemsize*msg.size*values.nmsg*num_transmitters_sending_to_me*8/float(interval)/1e9
    rate = msg.itemsize*msg.size*numCompletionsTotal*num_transmitters_sending_to_me*8/float(interval)/1e9
    log.info(f'Rank {rank} receiver data rate is {rate} Gbps')
    log.info(f'message loss rate is {numMissingTotal/float(numMessagesTotal)}')
        
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
    # Get messages now
    msg = np.zeros(values.msg_size)

    if values.method == 'mpi':
        for imsg in range(values.nmsg):
            log.debug(f'Sending msg {imsg} from transmitter {transmitter_rank} to receiver {receiver_rank}')
            world.Send(msg+imsg, dest=receiver_rank, tag=transmitter_rank)

    if values.method == 'rdma':
        rdma_receiver_info = world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        log.info(f'Got rdma receiver info {rdma_receiver_info} from rdma receiver with rank={status.Get_source()} tag={status.Get_tag()}')
        
        rdma_receiver_psn = rdma_receiver_info['psn']
        rdma_receiver_qpn = rdma_receiver_info['qpn']
        rdma_receiver_gid = rdma_receiver_info['gid']
        rdma_receiver_lid = rdma_receiver_info['lid']
        
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
                                         messageDelayTimeSend,
                                         rdmaDeviceName,
                                         rdmaPort,
                                         gidIndex,
                                         metricURL,
                                         numMetricAveraging)
        
        rdma_transmitter_psn = rdma_transmitter.getPacketSequenceNumber()
        rdma_transmitter_qpn = rdma_transmitter.getQueuePairNumber()
        rdma_transmitter_gid = np.frombuffer(rdma_transmitter.getGidAddress(), dtype=np.uint8)
        rdma_transmitter_lid = rdma_transmitter.getLocalIdentifier()
        rdma_transmitter_info = {'rank':rank, 'psn':rdma_transmitter_psn, 'qpn': rdma_transmitter_qpn,
                            'gid': rdma_transmitter_gid, 'lid':rdma_transmitter_lid}
        
        log.info(f'Sending rdma transmitter info {rdma_transmitter_info} to a rdma receiver with rank {receiver_rank}')
        world.send(rdma_transmitter_info, dest=int(receiver_rank), tag=1)
        
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

        rdma_buffer = []
        for iblock in range(numMemoryBlocks):
            rdma_memory = rdma_transmitter.get_memoryview(iblock)
            rdma_buffer.append(np.frombuffer(rdma_memory, dtype=np.int16))
            
        world.Barrier()
        numCompletionsTotal = 0
        while numCompletionsTotal < values.nmsg:

            # The setup here will not be strightforward
            #rdma_memory = rdma_transmitter.get_memoryview(0)
            #rdma_buffer = np.frombuffer(rdma_memory, dtype=np.int16) 
            #rdma_buffer[:] = 1
            #print(f'BEFORE:\tThe first {ndataPrint} data message from rdma transmitter {transmitter_rank} to receiver {receiver_rank} is {rdma_buffer[0:ndataPrint]}')
            
            rdma_transmitter.issueRequests()
            rdma_transmitter.waitRequestsCompletion()
            rdma_transmitter.pollRequests()
            numCompletionsFound = rdma_transmitter.get_numCompletionsFound()
            numCompletionsTotal += numCompletionsFound

            #print(f'{numCompletionsTotal} on transmitter {transmitter_rank}, the last numCompletionsFound is {numCompletionsFound}')
            workCompletions = rdma_transmitter.get_workCompletions()

            for i in range(numCompletionsFound):
                index = workCompletions[i].wr_id
            
                # Get data for buffer regions
                block_index = index//numContiguousMessages
                #rdma_memory = rdma_transmitter.get_memoryview(block_index)
                #rdma_buffer = np.frombuffer(rdma_memory, dtype=np.int16)
            
                # now it is data for each message
                message_index = index%numContiguousMessages
                data_start_index = int(message_index*messageSize*8//16) # because messageSize is in bytes, but we are using uint16_t, which is 2 bytes

                #print(f"why here we have problem {i}")
                #print(f'The first {ndataPrint} data of message from transmitter {transmitter_rank} to receiver {receiver_rank} is {rdma_buffer[data_start_index:data_start_index+ndataPrint]}')
                
            #time.sleep(1)
        
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nrx', type=int, help='Number of receivers', default=1)
    parser.add_argument('--nlink', type=int, help='Number of transmit links (2x number of cards)', default=1)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--nmsg', type=int, default=10)
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

    # Setup nmsage to a reasonable number when we use rdma method
    # ignore the passed nmsg
    # nmsg will only be used at 'mpi' mode
    if values.method == 'rdma':
        values.nmsg = numTotalMessages
        
    # ranks < nrx are receivers
    # ransk >= nrx are transmitter
    log.info(f"World rank {rank} size {size}")

    if rank < values.nrx:
        be_receiver(values)
    else:
        be_transmitter(values)

# mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --nmsg 2000
if __name__ == '__main__':
    _main()
