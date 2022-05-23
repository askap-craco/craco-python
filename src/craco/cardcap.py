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
import warnings

import roce_packetizer
from craft.fitswriter import FitsTableWriter

from rdma_transport import RdmaTransport
from rdma_transport import runMode
from rdma_transport import ibv_wc
from rdma_transport import ibv_wc_status
import rdma_transport
import socket
from craft.cmdline import strrange
from craco.epics.craco import Craco as CracoEpics
from astropy.time import Time

log = logging.getLogger(__name__)

FINE_CHANBW = 1.0*32./27./64. # MHz
FINE_TSAMP = 1.0/FINE_CHANBW # Microseconds

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
            ('version','<u4'), # Version
            ('zero3','<u4') # word 8
]
            

def get_single_packet_dtype(nbl: int, enable_debug_hdr: bool, sum_pols: bool=False):
    '''
    Gets the numpy dtype of a single ROCE packet sent from the correlator.

    if enable_debug_hdr the debug dtype is added
    The data shape is (nint, nbl, npol, 2)

    nint is the number of time integrations
    nbl is the number of baselines
    npol is the number of polarisations
    2 is for complex

    Whether sum_pols is true or false, a single packet always contiains 2xnbl entries. This is how
    John's firmware works.

    if sum_pols is True, npol=1, nint=2
    if sum_pols is False, npol=2, nint=1
    
    '''

    if enable_debug_hdr:
        dtype = debughdr[:]
    else:
        dtype = []

    if sum_pols:
        npol = 1
        nint = 2
    else: # dual-pol mode
        npol = 2
        nint = 1
                     
    dtype.append(('data', '<i2', (nint, nbl, npol, 2)))
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
    assert 1 <= shelf <= 8
    assert 1 <= card <= 12
    assert 0 <= fpga <= 6
    mac = bytearray(6)
    # Anything with teh 2nd nibble as 2, 6, A, E is Locally Administered https://serverfault.com/questions/40712/what-range-of-mac-addresses-can-i-safely-use-for-my-virtual-machines
    mac[0] = 0xaa 
    mac[1] = 0xce
    mac[2] = 0xf6
    mac[3] = shelf
    mac[4] = card

    # Every 2nd FPGA is conneted to the same fibre
    mac[5] = fpga % 2 
    
    return mac

hdr_size = 36*17 # words

class FpgaCapturer:
    def __init__(self, ccap, fpga):
        self.ccap = ccap
        values = ccap.values

        mode = runMode.RECV_MODE
        device = values.device
        rdmaPort = values.port
        gidIndex = values.gid_index
        msg_size = self.ccap.msg_size
        npacket_per_msg = self.ccap.npacket_per_msg
        num_blks = values.num_blks
        num_cmsgs = values.num_cmsgs
        nmsg = 1000000
        shelf = values.block
        card = values.card
        send_delay = 0
        num_cmsgs = num_cmsgs
        rx = RdmaTransport(mode,
                           msg_size,
                           num_blks,
                           num_cmsgs,
                           nmsg,
                           send_delay,
                           device,
                           rdmaPort,
                           gidIndex)

        self.rx = rx
        rx.checkImmediate = False
        psn = rx.getPacketSequenceNumber()
        qpn = rx.getQueuePairNumber()
        gid = np.frombuffer(rx.getGidAddress(), dtype=np.uint8)
        gids = '-'.join([f'{x:d}' for x in gid])
    
        log.info('RX PSN %d QPN %d =0x%x GID: %s %s', psn, qpn, qpn, mac_str(gid), gids)
        dst_mac = bytes(ipv6_to_mac(gid))
        src_mac = bytes(src_mac_of(shelf, card, fpga))
        src_gid = np.frombuffer(mac_to_ipv6(src_mac), dtype=np.uint8)
        log.info('Src MAC %s Dst MAC %s', mac_str(src_mac), mac_str(dst_mac))
        log.info('Src GID %s Dst GID %s', mac_str(src_gid), mac_str(gid))
       
        hdr = roce_packetizer.roce_header()
        hdr.setup(src_mac, dst_mac, qpn, psn)
        hbytes = bytes(hdr.to_array(True))
        print('Header is')
        hstring = ' '.join('0x{:02x}'.format(x) for x in hbytes)
        print(hstring)
        with open(f'header_fpga{fpga}.txt', 'w') as fout:
            fout.write(hstring)
        
        with open(f'header_{fpga}.bin','wb') as fout:
            fout.write(hbytes)


        hint = np.zeros(hdr_size, dtype=np.uint32)
        beam0_header = list(np.frombuffer(hbytes, dtype=np.uint32))
        hint[:len(beam0_header)] = beam0_header
        print(list(map(hex,beam0_header)))

                
        rdma_buffers = []
        self.rdma_buffers = rdma_buffers
        for iblock in range(num_blks):
            m = rx.get_memoryview(iblock)
            mnp = np.frombuffer(m, dtype=ccap.packet_dtype)
            mnp[:] = 0
            mnp.shape = (num_cmsgs, npacket_per_msg)
            log.debug('iblock %d shape=%s size=%d', iblock, mnp.shape, mnp.itemsize)
            rdma_buffers.append(mnp)

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
        log.info(f"Setup RDMA for fpga {fpga}")

        ccap.ctrl.set_roce_header(shelf, card, fpga, hint)
        self.curr_imm = 0
        self.nmiss = 0
        self.total_completions = 0
        self.total_missing = 0


    def issue_requests(self):
        rx = self.rx
        rx.issueRequests()
        log.info(f"Requests issued Enqueued={rx.numWorkRequestsEnqueued} missing={rx.numWorkRequestsMissing} total={rx.numTotalMessages} qloading={rx.currentQueueLoading} min={rx.minWorkRequestEnqueue} region={rx.regionIndex}")

    def write_data(self, w):
        rx = self.rx
        msg_size = self.ccap.msg_size
        rx.waitRequestsCompletion()
        rx.pollRequests()
        ncompletions = rx.get_numCompletionsFound()
        nmissing = rx.get_numMissingFound()
        completions = rx.get_workCompletions()
        self.total_completions += ncompletions
        self.total_missing += nmissing
        assert ncompletions == len(completions)
        num_cmsgs = self.ccap.num_cmsgs
        #print(f'\rCompletions={ncompletions} {len(completions)}')
        beam = self.ccap.values.beam
        for c in completions:
            #assert c.status == ibv_wc_status.IBV_WC_SUCCESS
            index = c.wr_id
            nbytes = c.byte_len
            assert nbytes == msg_size, f'Unexpected messages size. Was={nbytes} expected={msg_size}'
            immediate = c.imm_data
            immediate = socket.ntohl(immediate)
            if immediate != self.curr_imm:
                missed  = (immediate - self.curr_imm)
                self.nmiss += missed
                print(f'Missed {missed} nmiss={self.nmiss} nbytes={nbytes}')

            self.curr_imm = immediate+1
                    
            # Get data for buffer regions
            block_index = index//num_cmsgs
                
            # now it is data for each message
            message_index = index%num_cmsgs
            d = self.rdma_buffers[block_index][message_index]
            print(d.dtype)
            print(f"version={d[0]['version']} d={d[0]['nprod']}")
            print(d['version'])
            #w.write(d[:nbytes]) # write to fitsfile
            if beam is not None:
                mask = d['beam_number'] == beam 
                d = d[mask]

            w.write(d) # write to fitsfield
            
        rx.issueRequests()


class CardCapturer:
    def __init__(self, values):
        rdma_transport.setLogLevel(rdma_transport.logType.LOG_DEBUG)
        log.info('Starting card capture %s', values)
        self.values = values
        mode = runMode.RECV_MODE
        device = values.device
        rdmaPort = values.port
        gidIndex = values.gid_index
        # In mike's test setup he sends beam0, channel 155, samples 0,1,2,3. Samp=0 is FIRST, 1,2 are MIDDLE and 3 is LAST with immediate.
        # I.e. there are 4 packets per "message"
        nbeam = 36
        nchan = 4
        nsamp_per_frame = 2048
        nsamp_per_integration = values.samples_per_integration
        nint_per_frame = nsamp_per_frame // nsamp_per_integration
        tsamp = nsamp_per_integration * FINE_TSAMP # microsec
        include_autos = True
        nant = 30
        nbl = nant*(nant-1)//2

        if include_autos:
            nbl += nant

        npacket_per_msg= nbeam*nchan*nint_per_frame

        if values.pol_sum:
            npacket_per_msg //= 2
            
        enable_debug_header = values.debug_header
        packet_dtype = get_single_packet_dtype(nbl, enable_debug_header, values.pol_sum)
        self.packet_dtype = packet_dtype
        msg_size = packet_dtype.itemsize*npacket_per_msg
        self.msg_size = msg_size
        self.npacket_per_msg = npacket_per_msg
        num_blks = values.num_blks
        num_cmsgs = values.num_cmsgs
        nmsg = 1000000
        shelf = values.block
        card = values.card
        send_delay = 0
        self.num_cmsgs = num_cmsgs
        
        log.info(f'Listening on {device} port {rdmaPort} for msg_size={msg_size} {num_blks} {num_cmsgs} {nmsg} npacket_per_msg={npacket_per_msg} nbl={nbl} dtype={packet_dtype}')
        
        fpgaMask = 0x3f
        enMultiDest = False # fixed
        enPktzrDbugHdr = enable_debug_header
        enPktzrTestData = values.enable_test_data
        lsbPosition = values.lsb_position
        # lsbposition=0 is bits 15-0 from 27-bit accumulator ouput.
        # lsbPosition=1 is bits 16-1
        # ..
        # lsbPosition=11 = bits 27-11 (see discussion from john on mattermost)
        assert 0 <= lsbPosition <= 11, 'Unsupported LSB position'
        sumPols = 1 if values.pol_sum else 0
        integSelectMap = {16:0, 32:1, 64:2}
        # -- "00" : 16 samples = 864us
        # -- "01" : 32 samples = 1,728us (default)
        # -- "10" : 64 samples = 3,456us
        integSelect = integSelectMap[values.samples_per_integration]
        self.byteswap = False

        # configure CRACO on all FPGAS
        logging.info('Starting CRACO via EPICS')
        ctrl = CracoEpics(values.prefix+':')
        card_freqs = ctrl.get_channel_frequencies(shelf, card).reshape(6,4,9) # (6 fpgas, 4 coarse channels, 9 fine channels)
        fdiffs = card_freqs[:, :, 1:] - card_freqs[:, :, :-1]
        if fdiffs[0,0,0] == 0:
            warnings.warn("Invalid channel frequencies")

        #assert np.all(fdiffs == fdiffs[0,0,0]), f'Unexpected fdiffs {fdiffs[0,0,0]}'
        # fine channels are summed
        avg_freqs = card_freqs.mean(axis=2) # shape is (6 fpgas, 4 coarse channels)
        fch1 = avg_freqs[0,0]
        fpga_foff = avg_freqs[1:,:] - avg_freqs[:-1,:]
        coarse_foff = avg_freqs[:,1:] - avg_freqs[:,:-1]

        print('FPGA FOFF', fpga_foff)
        print('Coarse FOFF', coarse_foff)
        print(fpga_foff == fpga_foff[0,0])
        print(fpga_foff - fpga_foff[0,0])

        assert np.all(fpga_foff - fpga_foff[0,0] < 1e-6), 'FPGA frequency offset not always the same'
        assert np.all(coarse_foff - coarse_foff[0,0] < 1e-6), 'Coarse frequency offset not always the same'


        hdr = {}
        
        now = Time.now()
        now.format = 'fits'
        hdr['MJDNOW'] = now.mjd
        hdr['JDNOW'] = now.jd
        hdr['UTCNOW'] = str(now)
        hdr['FCH1'] = fch1
        hdr['FOFFFPGA'] = fpga_foff[0,0]
        hdr['FOFFCHAN'] = coarse_foff[0,0]
        hdr['SHELF'] = shelf
        hdr['CARD'] = card
        hdr['ARGV'] = ' '.join(sys.argv)
        hdr['LSBPOS'] = lsbPosition
        hdr['POLSUM'] = int(values.pol_sum)
        hdr['DUALPOL'] = int(values.dual_pol)
        hdr['INTEGSE'] = integSelect
        hdr['SAMPINT'] = values.samples_per_integration
        hdr['PREFIX'] = values.prefix
        hdr['TESTDATA'] = int(values.enable_test_data)
        hdr['DEBUGHDR'] = int(enable_debug_header)
        hdr['DEVICE'] = device
        hdr['RDMAPORT'] = rdmaPort
        hdr['GIDINDEX'] = gidIndex
        hdr['TSAMP'] = tsamp/1e6
        hdr['BEAM'] = -1 if values.beam is None else values.beam
        hdr['FPGA'] = str(values.fpga)

        ctrl.stop()
        ctrl.configure(fpgaMask, enMultiDest, enPktzrDbugHdr, enPktzrTestData, lsbPosition, sumPols, integSelect)
        self.ctrl = ctrl

        log.info(f'Shelf {shelf} card {card} Receiving data from {len(values.fpga)} fpgas: {values.fpga}')

        #self.clear_headers()
        self.fpga_cap = [FpgaCapturer(self, fpga) for fpga in values.fpga]

        # start CRACO (enabling packetiser, craco subsystem and firing event)
        try:
            w = FitsTableWriter(values.outfile, self.packet_dtype, self.byteswap, hdr)
            ctrl.start_shelf(shelf, [card])
            log.info('Started OK. Now saving data')
            self.save_data(w)
        except KeyboardInterrupt as e:
            print(f'Got an exceptio  {e}')
        finally:
            ctrl.stop()
            w.close()

    def clear_headers(self):
        zerohdr = np.zeros(hdr_size, dtype=np.uint32)
        for c in range(1,12):
            for f in range(1,6):
                self.ctrl.set_roce_header(self.values.block, c, f, zerohdr)
                

    def save_data(self, w):

        for fpga in self.fpga_cap:
            fpga.issue_requests()

        nblk = 0
        while True:
            for fpga in self.fpga_cap:
                fpga.write_data(w)

            nblk += 1
            if nblk >= self.values.num_msgs:
                break

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Raw download and saving ROCE data from correlator cards', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d','--device', help='RDMA device', default='mlx5_0')
    parser.add_argument('-p','--port', help='RDMA port', type=int, default=1)
    parser.add_argument('-g','--gid-index', help='RDMA GID index', type=int, default=0)
    parser.add_argument('-n','--num-blks', help='Number of ringbuffer slots', type=int, default=16)
    parser.add_argument('-c','--num-cmsgs', help='Numebr of messages per slot', type=int, default=1)
    parser.add_argument('--num-msgs', help='Total number of messages to download before quitting', default=-1, type=int)
    parser.add_argument('-e','--debug-header', help='Enable debug header', action='store_true', default=False)
    parser.add_argument('--prompt', help='Prompt for PSN/QPN/GID from e.g. rdma-data-transport/recieve -s', action='store_true', default=False)
    parser.add_argument('-f', '--outfile', help='Data output file')
    parser.add_argument('-b','--block',help='Correlator block to talk to', default=7, type=int) 
    parser.add_argument('-a','--card', help='Card range to talk to', default=1, type=int)
    parser.add_argument('-k','--fpga', help='FPGA range to talk to', default='1-6', type=strrange)
    parser.add_argument('--prefix', help='EPICS Prefix ma or ak', default='ma')
    parser.add_argument('--enable-test-data', help='Enable test data mode on FPGA', action='store_true', default=False)
    parser.add_argument('--beam', default=None, type=int, help='Which beam to save (default=all)')
    parser.add_argument('--lsb-position', help='Set LSB position in CRACO quantiser (0-11)', type=int, default=11)
    parser.add_argument('--samples-per-integration', help='Number of samples per integration', type=int, choices=(16, 32, 64), default=32)


    pol_group = parser.add_mutually_exclusive_group(required=True)
    pol_group.add_argument('--pol-sum', help='Sum pol mode', action='store_true')
    pol_group.add_argument('--dual-pol', help='Dual pol mode', action='store_true')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    CardCapturer(values)
    

if __name__ == '__main__':
    _main()
