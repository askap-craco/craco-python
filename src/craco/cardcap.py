#!/usr/bin/env python
"""
captures CRACO data from correlator cards

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
import copy
import time
import socket

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
from astropy.io import fits
from craco import leapseconds
from craco.utils import ibc2beamchan
from craco.cardcapfile import * 


log = logging.getLogger(__name__)
hostname = socket.gethostname()

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


def uint8tostr(d):
    if isinstance(d, str):
        s = d
    else:
        s = ''.join(map(chr, d))

    return s.strip().replace('\x00','')

def get_indexes(nant):
    '''
    Returns a set of array indexs that can be used to index into baseline arrays
    assumign the way the correlator orders everythign (baseically, sensibly)
    One day if you have a more complex configuration than just antennas, this might
    need to be made more sophisticated.

    Returns 4-typple containing (products, revproducts, auto_products, cross_products
    where
    products: Array length=nbl, contains (a1, a2) where a1 is antenna1 and a2 is antenna2 (1 based)
    revproducts: dictionary length(nbl) keyed by tuple (a1, a2) and returns baseline index
    auto_products: length=nant array of which indices in teh correlation matrix contain autocorrelations
    cross_products: length=nbl array of which indices contain cross correlations
    '''
    
    products = []
    revproducts = {}
    auto_products = []
    cross_products = []
    idx = 0
    for a1 in range(1, nant+1):
        for a2 in range(a1, nant+1):
            products.append((a1,a2))
            revproducts[(a1,a2)] = idx
            if a1 == a2:
                auto_products.append(idx)
            else:
                cross_products.append(idx)
            
            idx += 1
              
    products = np.array(products, dtype=[('a1',np.int16), ('a2', np.int16)])

    return (products, revproducts, auto_products, cross_products)



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

def mac_to_ipv6(mac, vid=0xffff):
    '''
    Converts link-local IPV6 address to mac address
    :mac: Mac address bytes
    :vid: uint16 vland ID 
    See https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=af7bd463761c6abd8ca8d831f9cc0ac19f3b7d4b for vlan calc
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

    if vid != 0xffff: # add vlan tag
        ipv6[11] = vid >> 8
        ipv6[12] = vid & 0xff
    else:
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

def set_vlan(gid, vlan=883):
    '''
    No documentaiton anywhere, but apprently vlan is in bytes 12 and 13 of the GID
    See: https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet#Criticism
    And: https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet#Criticism
    '''
    assert vlan == 883

    # Vlan 883 is 0x0373
    # And the array is backwards
    # And I don't know if it's backwards
    # idx=0 is "byte 16"
    # idx=1 is byte 15
    # idx=2 is byte 14
    # idx=3 is byte 13
    # idx=4 is byte 12
    gid[3] = 0x73
    gid[4] = 0x03
    return gid

def get_net_dev_of_gid(device, port, gididx):
    # Look up this stuff
    # Documented here: https://docs.nvidia.com/networking/pages/viewpage.action?pageId=12013422
    #/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/
    # in particular
    # $ cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/0
    # enp175s0
    # and
    # $ cat /sys/class/net/enp175s0/address
    # 0c:42:a1:55:c1:ee
    ndev = f'/sys/class/infiniband/{device}/ports/{port}/gid_attrs/ndevs/{gididx}'
    with open(ndev, 'rt') as f:
        nd = f.read().strip()

    return nd

def get_mac_of_net_dev(netdev):
    macpath = f'/sys/class/net/{netdev}/address'
    with open(macpath, 'rt') as f:
        mac_string = f.read().strip()
        mac_bytes = bytes(map(lambda x: int(x, 16), mac_string.split(':')))

    return mac_bytes

hdr_size = 36*17 # words

class FpgaCapturer:
    def __init__(self, ccap, fpga):
        self.ccap = ccap
        self.fpga = fpga
        self.values = ccap.values
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
        # override packet sequence number
        rx.packetSequenceNumber = 0
        psn = rx.getPacketSequenceNumber()
        assert psn == 0
        
        qpn = rx.getQueuePairNumber()
        dst_gid = np.frombuffer(rx.getGidAddress(), dtype=np.uint8)
        gids = '-'.join([f'{x:d}' for x in dst_gid])
    
        log.info('RX PSN %d QPN %d =0x%x GID: %s %s', psn, qpn, qpn, mac_str(dst_gid), gids)
        #dst_mac = bytes(ipv6_to_mac(dst_gid))
        src_mac = bytes(src_mac_of(shelf, card, fpga))
        vid = 0x4373 # probably waht it should be
        #vid = 0xffff # thsi works if no trunks enabled and no vlans
        netdev = get_net_dev_of_gid(device, rdmaPort, gidIndex)
        dst_mac = get_mac_of_net_dev(netdev)
        src_gid = np.frombuffer(mac_to_ipv6(src_mac, vid), dtype=np.uint8)

        log.info('Src MAC %s Dst MAC %s netdev %s', mac_str(src_mac), mac_str(dst_mac), netdev)
        log.info('Src GID %s Dst GID %s', mac_str(src_gid), mac_str(dst_gid))
       
        hdr = roce_packetizer.roce_header()
        hdr.setup2(src_mac, dst_mac, src_gid.tobytes(), dst_gid.tobytes(), qpn, psn)
        hbytes = bytes(hdr.to_array(True))
        hstring = ' '.join('0x{:02x}'.format(x) for x in hbytes)
        with open(f'header_fpga{fpga}.txt', 'w') as fout:
            fout.write(hstring)
        
        #with open(f'header_{fpga}.bin','wb') as fout:
        #    fout.write(hbytes)

        hint = np.zeros(hdr_size, dtype=np.uint32)
        beam0_header = list(np.frombuffer(hbytes, dtype=np.uint32))
        hint[:len(beam0_header)] = beam0_header
                
        rdma_buffers = []
        self.rdma_buffers = rdma_buffers
        for iblock in range(num_blks):
            m = rx.get_memoryview(iblock)
            mnp = np.frombuffer(m, dtype=ccap.packet_dtype)
            mnp[:] = 0
            mnp.shape = ccap.msg_shape
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
        self.curr_imm = None
        self.curr_fid = None
        self.total_completions = 0
        self.total_missing = 0
        self.total_bytes = 0
        self.craco63_last = 0
        self.max_ncompletions = 0

    def issue_requests(self):
        rx = self.rx
        rx.issueRequests()
        log.debug(f"Requests issued Enqueued={rx.numWorkRequestsEnqueued} missing={rx.numWorkRequestsMissing} total={rx.numTotalMessages} qloading={rx.currentQueueLoading} min={rx.minWorkRequestEnqueue} region={rx.regionIndex}")

    def get_data(self, wait=True):
        '''
        Returns an iterator for all the completions
        Iterator is (frame_id, data) pairs
        '''
        rx = self.rx
        msg_size = self.ccap.msg_size

        start_sleep_ns = 0#time.time_ns()
        #time.sleep(1)
        stop_sleep_ns = 0# time.time_ns()
        if wait:
            rx.waitRequestsCompletion()

        finish_wait_ns = 0 #time.time_ns()
        sleep_ns = stop_sleep_ns - start_sleep_ns
        wait_ns = finish_wait_ns - stop_sleep_ns

        rx.pollRequests()
        ncompletions = rx.get_numCompletionsFound()
        
        #print(f'{hostname} {self.values.block}/{self.values.card}/{self.fpga} Slept for {sleep_ns/1e6}ms. Waited {wait_ns/1e6}ms for ncompletions={ncompletions}')
        #nmissing = rx.get_numMissingFound()
        completions = rx.get_workCompletions()
        self.curr_packet_time = time.process_time()
        if self.total_completions == 0 and ncompletions > 0:
            self.first_packet_time = self.curr_packet_time

        assert ncompletions == len(completions)
        num_cmsgs = self.ccap.num_cmsgs
        #print(f'\rCompletions={ncompletions} {len(completions)}')
        beam = self.ccap.values.beam
        if ncompletions > self.max_ncompletions:
            log.critical(f'{hostname} {self.values.block}/{self.values.card}/{self.fpga} increased completions {self.max_ncompletions}->{ncompletions}')
            self.max_ncompletions = ncompletions
            
        d = None # in case something quits before it's set. We return None so we don't get an unboundLocalError

        # number of of packets per message (1 message = 1 completion)
        # should be either 1 or 144
        npacket_per_message = int(self.ccap.nbeam_per_message * self.ccap.nchan_per_message)
        assert npacket_per_message == 1 or npacket_per_message == NCHAN*NBEAM
        
        for c in completions:
            index = c.wr_id
            nbytes = c.byte_len
            assert nbytes == msg_size, f'Unexpected messages size. Was={nbytes} expected={msg_size}'
            immediate = c.imm_data

            immdiff = NSAMP_PER_FRAME # Immediate value should increase by 2048 samples for every ... frame
            ibc = self.total_completions*npacket_per_message % (NCHAN*NBEAM) # which bemchan index we're up to.

            if self.curr_imm is None: # set to first current value
                expected_immediate = immediate
            elif ibc == 0: # increment expected immediate if we're on the first packet of the next frame
                expected_immediate = (self.curr_imm + immdiff) % (1<<32)
            else:
                expected_immediate = self.curr_imm # in flush-on-beam mode we get lots of packets with the same immediate. Cool@

            diff  = (immediate - expected_immediate)
            # Get data for buffer regions
            block_index = index//num_cmsgs
            # now it is data for each message
            message_index = index%num_cmsgs
            d = self.rdma_buffers[block_index][message_index]
            fid = d['frame_id'][0, 0] # Frame ID is the frame ID of the first sample of the intgration, according ot John Tuthill
            if self.curr_fid is None:
                fid_diff = 0
            else:
                fid_diff = fid - self.curr_fid

            expected_beamchan = ibc2beamchan(ibc)

            #print(immediate, expected_immediate, diff, fid, self.curr_fid, fid_diff, ibc, expected_beamchan, npacket_per_message)
            # need to fix this logic for flush-on-beam
            if immediate != expected_immediate:
                self.total_missing += diff
                log.critical(f'{hostname} {self.values.block}/{self.values.card}/{self.fpga} MISSED PACKET imm={immediate}={hex(immediate)} expected={expected_immediate} Diff={diff} fid={fid}={hex(fid)} fid_diff={fid_diff}   nmiss={self.total_missing} nbytes={nbytes} qloading={rx.currentQueueLoading}')
            

            self.curr_imm = immediate
            self.curr_fid = fid
            self.total_completions += 1
            self.total_bytes += nbytes

            if self.ccap.values.workaround_craco63:
                d['data'] = np.roll(d['data'], shift=-1, axis=0)

            if self.ccap.values.tscrunch != 1:
                # BUG: When tscrunch != and polsum, we average over the packest, but not inside the packet, by accident.
                # THis will need to be fixed, but no time now. It's Xmas!
                # OK this is slow, but it works
                dout = np.empty(d.shape[0], dtype=d.dtype)
                for field in ('frame_id','bat','beam_number','sample_number','channel_number','fpga_id','nprod','flags','zero1','version','zero3'):
                    dout[field] = d[field][:,0]

                dout['data'] = d['data'].mean(axis=1, dtype=np.float32).astype(np.int16)
                d = dout

            if beam is not None:
                assert 0 <= beam < 36, f'Invalid beam {beam}'
                mask = d['beam_number'] == beam 
                d = d[mask]

            yield fid, d

    def write_data(self, w):
        for fid, d in self.get_data(): # loop through completions
            if w is not None and d is not None:
                w.write(d) # write to fitsfile

        self.issue_requests()

    def packet_iterator(self):
        nblk = 0
        while nblk < self.values.num_msgs:
            for fid, d in self.get_data(): # loop through completions
                yield fid, d
                nblk += 1
                if nblk >= self.values.num_msgs:
                    break

            self.issue_requests()


    def __del__(self):
        log.info(f'Deleting RX for card {self.ccap.values.card}  FPGA {self.fpga}')
        del self.rx
        del self.rdma_buffers

class CardCapturer:
    def __init__(self, values, primary=False, pvcache={}):
        rdma_transport.setLogLevel(rdma_transport.logType.LOG_DEBUG)
        log.info('Starting card capture %s', values)
        self.values = values
        self.primary = primary
        mode = runMode.RECV_MODE
        device = values.device
        rdmaPort = values.port
        gidIndex = values.gid_index
        # In mike's test setup he sends beam0, channel 155, samples 0,1,2,3. Samp=0 is FIRST, 1,2 are MIDDLE and 3 is LAST with immediate.
        # I.e. there are 4 packets per "message"
        if values.flush_on_beam:
            nbeam_per_msg = 1
            nchan_per_msg = 1
        else:
            nbeam_per_msg = NBEAM
            nchan_per_msg = NCHAN
        

        nsamp_per_frame = 2048
        nsamp_per_integration = values.samples_per_integration
        nint_per_frame = nsamp_per_frame // nsamp_per_integration
        tsamp = nsamp_per_integration * FINE_TSAMP # microsec
        include_autos = True
        nant = 30
        nbl = nant*(nant-1)//2

        if include_autos:
            nbl += nant

        if values.pol_sum: # if polsum is enabled, we get 2 integrations per set of debug headers
            nint_per_packet = 2
            #assert values.tscrunch == 1, 'Dont support polsum and tscrunch - we make a mistake on the tscrunching'
            warnings.warn('Dont support polsum and tscrunch - we make a mistake on the tscrunching')
        else:
            nint_per_packet = 1

        nintpacket_per_frame = nint_per_frame // nint_per_packet # number of integrations per frame
        npacket_per_msg= nbeam_per_msg*nchan_per_msg*nintpacket_per_frame
            
        enable_debug_header = values.debug_header
        packet_dtype = get_single_packet_dtype(nbl, enable_debug_header, values.pol_sum)
        self.packet_dtype = packet_dtype
        msg_size = packet_dtype.itemsize*npacket_per_msg
        num_blks = values.num_blks
        num_cmsgs = values.num_cmsgs

        self.msg_size = msg_size
        self.npacket_per_msg = npacket_per_msg
        self.nbeam_per_message = nbeam_per_msg
        self.nchan_per_message = nchan_per_msg
        self.nint_per_frame = nint_per_frame
        self.nint_per_packet = nint_per_packet
        self.nintpacket_per_frame = nintpacket_per_frame
        self.nintout_per_frame = nintpacket_per_frame // self.values.tscrunch

        assert self.values.tscrunch == 1 or self.values.tscrunch * self.values.samples_per_integration == nsamp_per_frame, 'Invalid tscrunch - it must be 1 or multiply SPI to 2048'
        self.msg_shape = (num_cmsgs, self.nbeam_per_message*self.nchan_per_message, self.nintpacket_per_frame)
        self.out_shape = (self.nbeam_per_message*self.nchan_per_message, self.nintout_per_frame)
        nmsg = 1000000
        shelf = values.block
        card = values.card
        send_delay = 0
        self.num_cmsgs = num_cmsgs
        

        fpgaMask = values.fpga_mask
#        for ifpga in range(6):
#            if (ifpga + 1) in values.fpga:
#                bit = 1
#            else:
#                bit = 0
#
#            fpgaMask <<= 1
#            fpgaMask |= bit
            
        log.info(f'Listening on {device} port {rdmaPort} for msg_size={msg_size} {num_blks} {num_cmsgs} {nmsg} npacket_per_msg={npacket_per_msg} nbl={nbl} dtype={packet_dtype} fpgaMask=0x{fpgaMask:x}')

        flushOnBeam = values.flush_on_beam
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
        ctrl = CracoEpics(values.prefix+':', pvcache)
        self.ctrl = ctrl
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

        assert np.all(fpga_foff - fpga_foff[0,0] < 1e-6), 'FPGA frequency offset not always the same'
        assert np.all(coarse_foff - coarse_foff[0,0] < 1e-6), 'Coarse frequency offset not always the same'

        syncbat = ctrl.read('F_syncReset:startBat_O')

        hdr = {}
        self.hdr = hdr
        
        now = Time.now()
        now.format = 'fits'
        dtaiutc = leapseconds.dTAI_UTC_from_utc(now.to_datetime()).seconds
        dtaiutc2 = (now.tai.datetime - now.datetime).seconds

        dspversion = uint8tostr(ctrl.read(f"acx:s{shelf:02d}:S_corFpgaVersion:val"))
        iocversion = uint8tostr(ctrl.read(f'acx:s{shelf:02d}:version'))

        hdr['NANT'] = (nant, 'Number of antennas')
        hdr['NBL'] = (nbl, 'Number of baselines')
        hdr['AUTOS'] = (include_autos, 'T if autocorrelations are included')
        hdr['NOWMJD'] = (now.mjd, 'UTC MJD at file creation time')
        hdr['NOWJD'] = (now.jd, 'UTC JD at file creation time')
        hdr['NOWTAI'] = (str(now.tai), 'TAI string at file creation time')
        hdr['NOWSTR'] = (str(now), 'UTC string at file creation time')
        hdr['NOWUNIX'] = (now.unix, 'Now in UNIX time')
        hdr['NOWUTAI'] = (now.unix_tai, 'Now in UNIX TAI')
        hdr['DTAIUTC'] = (dtaiutc, 'TAI-UTC at file creation time from leapseconds.py')
        hdr['DTAIUTC2'] = (dtaiutc2, 'TAI-UTC at file creation time from astropy')
        hdr['FCH1'] = (fch1, 'Frequency of fpga=0, channel=0 (MHz)')
        hdr['FOFFFPGA'] = (fpga_foff[0,0], 'Frequency offset per FPGA')
        hdr['FOFFCHAN'] = (coarse_foff[0,0], 'Frequency offset per FPGA channel')
        hdr['SHELF'] = (shelf, 'Correlator shelf/block')
        hdr['CARD'] = (card, 'Correlator card')
        hdr['ARGV'] = (' '.join(sys.argv), 'Cardcap command line arguments')
        hdr['LSBPOS'] = (lsbPosition, 'LSB position')
        hdr['POLSUM'] = (values.pol_sum, 'T if POLSUM enabled')
        hdr['DUALPOL'] = (values.dual_pol, 'T if DUAL Pol mode enabled')
        hdr['INTEGSE'] = (integSelect, 'Integration selection value sent to hardware')
        hdr['SAMPINT'] = (values.samples_per_integration, 'Number of 18kHz samples per CRACO integration')
        hdr['PREFIX'] = (values.prefix, 'EPICS prefix. ma=MATES, ak=ASKAP')
        hdr['TESTDATA'] = (values.enable_test_data, 'T if packetiser test data mode enabled (counting pattern)')
        hdr['DEBUGHDR'] = (enable_debug_header, 'T if debug header in packetiser is enabled')
        hdr['DEVICE'] = (device, 'Which network card was used')
        hdr['RDMAPRT'] = (rdmaPort, 'Which RDMA port was used for RoCE')
        hdr['GIDINDX'] = (gidIndex, 'Which GID index was used for RoCE')
        hdr['TSAMP'] = (tsamp/1e6, 'Sampling time for CRACO integrations (seconds)')
        hdr['BEAM'] = (-1 if values.beam is None else values.beam, 'Beam downloaded. -1 is all beams')
        hdr['FPGA'] = (str(values.fpga), 'FPGAs downloaded (comma separated, 1 based)')
        hdr['NFPGA'] = (len(values.fpga), 'Number of FPGAs downloaded')
        hdr['SYNCBAT'] = (syncbat, 'Hexadecimal BAT when frame ID was set to 0')
        hdr['DSPVER'] = (dspversion, 'Block DSP version')
        hdr['IOCVER'] = (iocversion, "IOC version for block")
        hdr['HOST'] = (hostname, 'Capture host name')
        hdr['MSG_SIZE'] = (msg_size, 'Number of bytes in a message')
        hdr['NPCKMSG'] =  (self.npacket_per_msg, 'Number of debug header packets per message')
        hdr['NBEAM'] = (NBEAM, 'Number of beams being downlaoded (always 36)')
        hdr['NCHAN'] = (NCHAN, 'Number of channels per FPGA (4)')
        hdr['NBEPMSG'] = (self.nbeam_per_message, 'Number of beams per ROCE message')
        hdr['NCHPMSG'] = (self.nchan_per_message, 'Number of channels per ROCE message')
        hdr['NTPFM'] = (self.nint_per_frame, 'Total number of integrations per frame')
        hdr['NTPKFM'] = (self.nintpacket_per_frame, 'Total number of packet integrations per frame')
        hdr['MSGSHAPE'] = (str(self.msg_shape), 'Shape of a message buffer')
        hdr['TSCRUNCH'] = (self.values.tscrunch, 'Tscrunch factor')
        hdr['OUTSHAPE'] = (str(self.out_shape), 'Shape of file output')
        hdr['NTOUTPFM'] = (self.nintout_per_frame, 'Number of output integraitons per frame, after tscrunch')
        hdr['FLUSHBM'] = (flushOnBeam, 'T if flush on beam is enabled')
        
        self.hdr = hdr
        if values.prefix != 'ma':
            self.pvhdr('md2:targetName_O', 'TARGET','Target name from metadata')
            self.pvhdr('md2:scanId_O', 'SCANID','Scan ID from metadata')
            self.pvhdr('md2:schedulingblockId_O', 'SBID','SBID rom metadata')
            self.pvhdr('F_options:altFirmwareDir_O', 'FWDIR', 'Alternate firmware directory')

        log.info(f'Shelf {shelf} card {card} Receiving data from {len(values.fpga)} fpgas: {values.fpga}')
        self.configure_args  = (fpgaMask, flushOnBeam, enPktzrDbugHdr, enPktzrTestData, lsbPosition, sumPols, integSelect)

        #self.clear_headers()
        self.fpga_cap = [FpgaCapturer(self, fpga) for fpga in values.fpga]

        if self.values.outfile:
            thedir = os.path.dirname(values.outfile)
            if len(thedir.strip()) > 0:
                os.makedirs(thedir, exist_ok=True)

            self.fitsout = FitsTableWriter(self.values.outfile, self.packet_dtype, self.byteswap, self.hdr)
        else:
            self.fitsout = None

        # send initial request
        for fpga in self.fpga_cap:
            fpga.issue_requests()

    def pvhdr(self, pvname, card, comment):
        try:
            v = self.ctrl.read(pvname)
            self.hdr[card] = (v, comment)
        except:
            self.hdr[card] = ('PVERROR',comment)

    def do_writing(self):
        values = self.values
        try:
            log.info('Started OK. Now saving data to %s', values.outfile)
            self.save_data()
        except KeyboardInterrupt as e:
            print(f'Closing due to ctrl-C')
        finally:
            #self.stop(wait=False)
            self.finish_time = time.process_time()

            self.report_stats()
            if self.fitsout is not None:
                self.fitsout.close()
                
            self.fitsout = None
            #ctrl.write(f'acx:s{shelf:02d}:evtf:craco:enable', 0, wait=False)
            self.stop()
                

    def start(self):
        # start CRACO (enabling packetiser, craco subsystem and firing event)
        if self.primary:
            #ctrl.start_shelf(shelf, [card])
            self.ctrl.start()

    def stop(self, wait=True):
        if self.primary:
            self.ctrl.stop()
            if wait:
                time.sleep(1) # wait for it to stop - takes 110ms

    def configure(self):
        if self.primary:
            self.ctrl.stop()
            self.ctrl.configure(*self.configure_args)

    def report_stats(self):
        log.info(f'Block {self.values.block} card {self.values.card} completions={self.total_completions} bytes={self.total_bytes} missing={self.total_missing} pps={self.packets_per_sec}')

    @property
    def packets_per_sec(self):
        return [f.total_completions/(f.curr_packet_time - f.first_packet_time) for f in self.fpga_cap]

    @property
    def total_completions(self):
        return [f.total_completions for f in self.fpga_cap]

    @property
    def total_bytes(self):
        return [f.total_bytes for f in self.fpga_cap]

    @property
    def total_missing(self):
        return [f.total_missing for f in self.fpga_cap]


    def __del__(self):
        if hasattr(self, 'fpag_cap'):
            for f in self.fpga_cap:
                del f
            
            del self.fpga_cap

        if hasattr(self, 'ctrl'):
            del self.ctrl

    def clear_headers(self):
        zerohdr = np.zeros(hdr_size, dtype=np.uint32)
        for c in range(1,12):
            for f in range(1,6):
                self.ctrl.set_roce_header(self.values.block, c, f, zerohdr)

    def save_data(self):
        nblk = 0
        while True:
            for fpga in self.fpga_cap:
                fpga.write_data(self.fitsout)

            nblk += 1
            if nblk >= self.values.num_msgs:
                break


def hexstr(s):
    return int(s, 16)

def dump_rankfile(values):
    from craco import mpiutil
    hosts = sorted(set(mpiutil.parse_hostfile(values.hostfile)))
    log.debug("Hosts %s", hosts)
    total_cards = len(values.block)*len(values.card)
    if values.max_ncards != None:
        total_cards = min(total_cards, values.max_ncards)

    nranks = total_cards*len(values.fpga)
    ncards_per_host = (total_cards)//len(hosts)
    if ncards_per_host == 0:
        ncards_per_host = 1

    
    #nranks_per_host = (nranks + len(hosts)) // len(hosts)
    nranks_per_host = ncards_per_host*6
    log.info(f'Spreading {nranks} ranks over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas ncards_per_host={ncards_per_host} nranks_per_host={nranks_per_host}')
    
    #assert nranks_per_host * len(hosts) >= nranks
    #from IPython import embed
    #embed()

    rank = 0
    cardno = 0
    with open(values.dump_rankfile, 'w') as fout:
        for block in values.block:
            for card in values.card:
                cardno += 1
                if values.max_ncards != None and cardno >= values.max_ncards + 1:
                    break
                
                for fpga in values.fpga:
                    hostidx = (rank // nranks_per_host) % len(hosts)
                    hostrank = rank % nranks_per_host
                    host = hosts[hostidx]
                    slot = 1 # fixed because both cards are on NUMA=1
                    # Put different FPGAs on differnt cores
                    evenfpga = fpga % 2 == 0
                    core = rank % 10
                    slot = 1
                    s = f'rank {rank}={host} slot={slot}:{core} # Block {block} card {card} fpga {fpga}\n'
                    fout.write(s)
                    rank += 1
                    
class MpiCardcapController:
    def __init__(self, comm, values, block_cards):
        rank = comm.Get_rank()
        numprocs = comm.Get_size()
        self.comm = comm
        self.rank = rank
        self.values = values
        
        primary = rank == 0

        # Before we start, we need to stop everything so nothign gets confused
        if rank == 0:
            ctrl = CracoEpics(values.prefix+':')
            ctrl.stop()
        else:
            ctrl = None

        self.ctrl = ctrl
            
        comm.Barrier()

        # only rank 0 gets EPICS data - otherwise lots of processes drown EPICS
        pvcache = None
        if rank == 0:
            # cache the values by reading
            syncbat = ctrl.read('F_syncReset:startBat_O')
            ctrl.read('md2:targetName_O')
            ctrl.read('md2:scanId_O')
            ctrl.read('md2:schedulingblockId_O')
            ctrl.read('F_options:altFirmwareDir_O')

            for shelf in values.block:
                dspversion = uint8tostr(ctrl.read(f"acx:s{shelf:02d}:S_corFpgaVersion:val"))
                iocversion = uint8tostr(ctrl.read(f'acx:s{shelf:02d}:version'))
                for card in values.card:
                    ctrl.get_channel_frequencies(shelf, card)

            pvcache = ctrl.cache
            log.debug(f'Cache contains {len(pvcache)} entries {pvcache}')

        # broadcast the cache to everyone

        pvcache = comm.bcast(pvcache, root=0)
        self.block_cards = block_cards

        if rank < len(block_cards):
            my_block, my_card, my_fpga =  block_cards[rank]
            my_values = copy.deepcopy(values)
            my_values.card = my_card
            my_values.block = my_block
            my_values.fpga = my_fpga

            devices = values.devices.split(',')
            #devices =['mlx5_0','mlx5_0']
            if len(my_fpga) == 1:
                devidx = my_fpga[0] % len(devices)
                #devidx = my_card % len(devices)
            else:
                devidx = my_card % len(devices)

            my_values.device = devices[devidx]

            if values.outfile is None:
                my_values.outfile = None
            else:
                my_values.outfile = values.outfile.replace('.fits','')
                fpga_names = ''.join([f'{f:d}' for f in my_fpga])
                my_values.outfile += f'_b{my_block:02d}_c{my_card:02d}+f{fpga_names}.fits'
                
            log.info(f'MPI CARDCAP: My rank is {rank}/{numprocs}. Downloaing card={my_values.card} block {my_values.block} fpga={my_values.fpga} to {my_values.outfile}')
            ccap = CardCapturer(my_values, primary, pvcache)
        else:
            log.info(f'Launched too may processes for me to do anything useful. Rank {rank} goin to sleep to get out of the way')
            my_values = None
            ccap = None

        self.ccap = ccap
        self.ctrl = ctrl

    def configure_and_start(self):
        '''
        :returns: The BAT that should be when the thing starts
        '''
        rank = self.rank
        values = self.values
        comm = self.comm
        ccap = self.ccap
        ctrl = self.ctrl

        comm.Barrier()
        start_bat = None

        if rank == 0:
            ccap.configure()
            # disable all cards, and enable only the ones we want
            #blk = values.block[0]
            #assert len(values.block) == 1, 'Cant start like that currently'

            # Enable only the cards we want.
            ctrl.enable_events_for_blocks_cards(values.block, values.card, values.max_ncards)
            # Normally do start() here but it would re-enable everything,
            # so just start this block
            #ctrl.start_block(blk)
            #ctrl.start_async(values.block, values.card) # starts async but I think does a better job of turnng stuff off
            ctrl.start()
            start_bat = ctrl.get_start_bat()
            log.info('Start bat is 0x%x=%d', start_bat, start_bat)

        self.start_bat = comm.bcast(start_bat, root=0)


        comm.Barrier()

        return self.start_bat
        

    def do_writing(self):
        if self.rank < len(self.block_cards):
            self.ccap.do_writing()

    def stop(self):
        ccap = self.ccap
        comm = self.comm
        rank = self.rank

        comm.Barrier()
        
        if rank == 0:
            ccap.stop()

        ncomplete = -1 if ccap is None else ccap.total_completions[0]
        completions = comm.gather(ncomplete)

        nmissing = -1 if ccap is None else ccap.total_missing[0]
        nmissing = comm.gather(nmissing)

        if rank == 0:
            log.info('Total completions=%s missing=%s', completions[:len(block_cards)], nmissing[:len(block_cards)])
        
    

def add_arguments(parser):
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d','--device', help='RDMA device', default='mlx5_0')
    parser.add_argument('-p','--port', help='RDMA port', type=int, default=1)
    parser.add_argument('-g','--gid-index', help='RDMA GID index', type=int, default=2)
    parser.add_argument('-n','--num-blks', help='Number of ringbuffer slots', type=int, default=16)
    parser.add_argument('-c','--num-cmsgs', help='Numebr of messages per slot', type=int, default=1)
    parser.add_argument('-N', '--num-msgs', help='Total number of messages to download before quitting', default=100, type=int)
    parser.add_argument('-e','--debug-header', help='Enable debug header', action='store_true', default=True) # need this to be true as lots of code expects it now. We'll probably keep it. the overhead isnt high I don't think
    parser.add_argument('--prompt', help='Prompt for PSN/QPN/GID from e.g. rdma-data-transport/recieve -s', action='store_true', default=False)
    parser.add_argument('-f', '--outfile', help='Data output file')
    parser.add_argument('-b','--block',help='Correlator block to talk to', default=7, type=strrange) 
    parser.add_argument('-a','--card', help='Card range to talk to', default=1, type=strrange)
    parser.add_argument('-k','--fpga', help='FPGA range to talk to', default='1-6', type=strrange)
    parser.add_argument('--devices', help='List of dievices to receive from, comman separated', default='mlx5_0,mlx5_1')
    parser.add_argument('--prefix', help='EPICS Prefix ma or ak', default='ak')
    parser.add_argument('--enable-test-data', help='Enable test data mode on FPGA', action='store_true', default=False)
    parser.add_argument('--beam', default=None, type=int, help='Which beam to save (default=all)')
    parser.add_argument('--lsb-position', help='Set LSB position in CRACO quantiser (0-11)', type=int, default=11)
    parser.add_argument('--samples-per-integration', help='Number of samples per integration', type=int, choices=(16, 32, 64), default=32)
    parser.add_argument('--tscrunch', help='Tscrunch by this factor before saving', type=int, default=1, choices=(1,2,4,8,16,32,64,128))
    parser.add_argument('--mpi', action='store_true', help='RunMPI version', default=False)
    parser.add_argument('--workaround-craco63', action='store_true', help='CRACO63 workaround', default=False)
    parser.add_argument('--fpga-mask', type=hexstr, help='(hex) FPGA mask for configuration', default=0x3f)
    parser.add_argument('--flush-on-beam', action='store_true', help='Flush a packet per beam, rather than per beamformer frame', default=False)
    parser.add_argument('--dump-rankfile', help='Dont run. just dump rankfile to this path')
    parser.add_argument('--hostfile', help='Hostfile to use to dump rankfile')
    parser.add_argument('--max-ncards', help='Set maximum number of cards to download 0=all', type=int, default=None)

    pol_group = parser.add_mutually_exclusive_group(required=True)
    pol_group.add_argument('--pol-sum', help='Sum pol mode', action='store_true')
    pol_group.add_argument('--dual-pol', help='Dual pol mode', action='store_true')


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Raw download and saving ROCE data from correlator cards', formatter_class=ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    parser.set_defaults(verbose=False)
    return parser

def _main():
    parser = get_parser()
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if values.dump_rankfile:
        dump_rankfile(values)
        sys.exit(0)

    if values.mpi:
        import mpi4py.rc
        mpi4py.rc.threads = False
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numprocs = comm.Get_size()

        # Assign 1 FPGA to every rank
        block_cards  = []
        procid = 0
        cardno = 0
        for blk in values.block:
            for crd in values.card:
                cardno += 1
                for fpga in values.fpga:
                    block_cards.append((blk, crd, [fpga]))
                    procid += 1
                    if procid > numprocs:
                        break


        controller = MpiCardcapController(comm, values,block_cards)
        controller.configure_and_start()
        controller.do_writing()
        controller.stop()
        
            
    else: # not mpi
        primary = True
        log.exception('MPI setup failed. Falling back to vanilla goodness of wonderment')
        my_values = values
        assert len(values.card) == 1 and len(values.block) ==1, 'Can only do one card at a time in vanilla mode'
        my_values.card = values.card[0]
        my_values.block = values.block[0]
        ccap = CardCapturer(my_values, primary)
        ccap.stop()
        ccap.configure()
        ccap.start()
        ccap.do_writing()
        



if __name__ == '__main__':
    _main()
