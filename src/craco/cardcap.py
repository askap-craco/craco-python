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
import roce_packetizer

from rdma_transport import RdmaTransport
from rdma_transport import runMode
from rdma_transport import ibv_wc
from rdma_transport import ibv_wc_status
import rdma_transport

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
debughdr = [('frame_id', '<u8'), # Words 1,2
            ('bat', '<u8'), # Words 3,4
            ('beam_number','<u1'), # Word 5
            ('sample_number','<u1'),
            ('channel_number','<u1'),
            ('fpga_id','<u1'), 
            ('nprod','<u2'),# Word 6
            ('flags','<u1'), # (sync_bit, last_packet, enable_debug_header, enable_test_data
            ('zero1','<u1'), # padding
            ('zero2','<u4'), # word 7
            ('zero3','<u4') # word 8
]
            

def get_single_packet_dtype(nprod, enable_debug_hdr):
    if enable_debug_hdr:
        dtype = debughdr[:]
    else:
        dtype = []
                     
    dtype.append(('data', '<i2', (nprod,)))
    return np.dtype(dtype)


def mac_str(mac):
    return ':'.join('{:02x}'.format(x) for x in mac)


def ipv6_to_mac(ipv6):
    '''
    Converts MAC to link-local IPV6 address
    '''
    assert len(ipv6) == 16
    mac = bytearray(6)

    mac[0] = ipv6[8] ^ 0b10; #/ invert bit 6
    mac[1] = ipv6[9]
    mac[2] = ipv6[10]
    mac[3] = ipv6[13]
    mac[4] = ipv6[14]
    mac[5] = ipv6[15]

    return mac

def mac_to_ipv6(mac):
    '''
    Converts link-local IPV6 address to mac address
    '''
    ipv6 = bytearray(16)

    ipv6[0] = 0xfe;
    ipv6[1] = 0x80;
    ipv6[2] = 0x00;
    ipv6[3] = 0x00;
    
    ipv6[4] = 0x00;
    ipv6[5] = 0x00;
    ipv6[6] = 0x00;
    ipv6[7] = 0x00;
    
    ipv6[8] = mac[0] ^ 0b10; #// Invert bit 6
    ipv6[9] = mac[1];
    ipv6[10] = mac[2];
    ipv6[11] = 0xff;
    
    ipv6[12] = 0xfe;
    ipv6[13] = mac[3];
    ipv6[14] = mac[4];
    ipv6[15] = mac[5];

    return ipv6


def src_mac_of(shelf, card, fpga):
    assert 0 <= shelf <= 256
    assert 0 <= card <= 256
    assert 0 <= fpga <= 256
    mac = bytearray(6)
    # Anything with teh 2nd nibble as 2, 6, A, E is Locally Administered https://serverfault.com/questions/40712/what-range-of-mac-addresses-can-i-safely-use-for-my-virtual-machines
    mac[0] = 0xaa 
    mac[1] = 0xce
    mac[2] = 0xf6
    mac[3] = shelf
    mac[4] = card
    mac[5] = 0# fpga - don't want fpgas to have different mac Addresses - this might confused the swith
    
    return mac
    
def run(values):
    rdma_transport.setLogLevel(rdma_transport.logType.LOG_DEBUG)
    mode = runMode.RECV_MODE
    device = values.device
    rdmaPort = values.port
    gidIndex = values.gid_index
    # In mike's test setup he sends beam0, channel 155, samples 0,1,2,3. Samp=0 is FIRST, 1,2 are MIDDLE and 3 is LAST with immediate.
    nbeam = 1
    nchan = 1
    nsamp = 4
    nprod = 1024
    enable_debug_header = True
    packet_dtype = get_single_packet_dtype(nprod, enable_debug_header)
    msg_size = packet_dtype.itemsize
    num_blks = values.num_blks
    num_cmsgs = values.num_cmsgs
    nmsg = 1000000
    shelf = 3
    card = 4
    fpga = 2
    send_delay = 0
    log.info(f'Listening on {device} port {rdmaPort} for {msg_size} {num_blks} {num_cmsgs} {nmsg}')

    rx = RdmaTransport(mode,
                       msg_size,
                       num_blks,
                       num_cmsgs,
                       nmsg,
                       send_delay,
                       device,
                       rdmaPort,
                       gidIndex)

    rx.checkImmediate = True
    psn = rx.getPacketSequenceNumber()
    qpn = rx.getQueuePairNumber()
    gid = np.frombuffer(rx.getGidAddress(), dtype=np.uint8)

    gids = '-'.join([f'{x:d}' for x in gid])

    log.info('RX PSN %d QPN %d =0x{%x} GID: %s %s', psn, qpn, qpn, mac_str(gid), gids)
    dst_mac = bytes(ipv6_to_mac(gid))
    src_mac = bytes(src_mac_of(shelf, card, fpga))
    src_gid = np.frombuffer(mac_to_ipv6(src_mac), dtype=np.uint8)
    log.debug('Src MAC %s Dst MAC %s', mac_str(src_mac), mac_str(dst_mac))
    
    hdr = roce_packetizer.roce_header()
    hdr.setup(src_mac, dst_mac, qpn, psn)
    hbytes = bytes(hdr.to_array(True))
    print('Header is')
    hstring = ' '.join('0x{:02x}'.format(x) for x in hbytes)
    print(hstring)
    with open('header.txt', 'w') as fout:
        fout.write(hstring)
        
    with open('header.bin','wb') as fout:
        fout.write(hbytes)

    
    rdma_buffers = []
    for iblock in range(num_blks):
        m = rx.get_memoryview(iblock)
        mnp = np.frombuffer(m, dtype=packet_dtype)
        mnp[:] = 0
        mnp.shape = (num_cmsgs,)
        log.debug('iblock %d shape=%s size=%d', iblock, mnp.shape, mnp.itemsize)
        rdma_buffers.append(mnp)

    outf = open('capture.dat', 'wb')
    #import ipdb; ipdb.set_trace()

    if values.prompt:
        psn = int(input("PSN: "))
        qpn = int(input("QPN: "))
        gidstr = input("GID: ")
        src_gid = np.array(list(map(int, gidstr.split("-"))), dtype=np.uint8)
        
    rx.setPacketSequenceNumber(psn)
    rx.setQueuePairNumber(qpn)
    rx.setGidAddress(src_gid)
    rx.setLocalIdentifier(0)
    rx.setupRdma(None)
    log.info("Setup RDMA")

    rx.issueRequests()
    log.info(f"Requests issued Enqueued={rx.numWorkRequestsEnqueued} missing={rx.numWorkRequestsMissing} total={rx.numTotalMessages} qloading={rx.currentQueueLoading} min={rx.minWorkRequestEnqueue} region={rx.regionIndex}")
    total_completions = 0
    total_missing = 0


    
    while True:
        rx.waitRequestsCompletion()
        rx.pollRequests()
        ncompletions = rx.get_numCompletionsFound()
        nmissing = rx.get_numMissingFound()
        completions = rx.get_workCompletions()
        total_completions += ncompletions
        total_missing += nmissing
        assert ncompletions == len(completions)
        print(f'\rCompletions={ncompletions} {len(completions)}')
        for c in completions:
            #assert c.status == ibv_wc_status.IBV_WC_SUCCESS
            index = c.wr_id
            nbytes = c.byte_len
            immediate = c.imm_data
            # Get data for buffer regions
            block_index = index//num_cmsgs
    
            # now it is data for each message
            message_index = index%num_cmsgs
            d = rdma_buffers[block_index][message_index]
            print(f'nbytes={nbytes} imm={immediate} 0x{immediate:x} index={index} {c.status}')
            print(f'{d.shape} {d.dtype}')
            #np.save(outf, d) # numpy way
            d.tofile(outf) # save raw bytes
            

        rx.issueRequests()
        log.info(f"Requests issued Enqueued={rx.numWorkRequestsEnqueued} missing={rx.numWorkRequestsMissing} total={rx.numTotalMessages} qloading={rx.currentQueueLoading} min={rx.minWorkRequestEnqueue} region={rx.regionIndex}")
        


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Raw download and saving ROCE data from correlator cards', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d','--device', help='RDMA device', default='mlx5_0')
    parser.add_argument('-p','--port', help='RDMA port', type=int, default=1)
    parser.add_argument('-g','--gid-index', help='RDMA GID index', type=int, default=0)
    parser.add_argument('-n','--num-blks', help='Number of ringbuffer slots', type=int, default=16)
    parser.add_argument('-c','--num-cmsgs', help='Numebr of messages per slot', type=int, default=1)
    parser.add_argument('-e','--debug-header', help='Enabel debug header', action='store_true', default=False)
    parser.add_argument('--prompt', help='Prompt for PSN/QPN/GID from e.g. rdma-data-transport/recieve -s', action='store_true', default=False)

    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(values)
    

if __name__ == '__main__':
    _main()