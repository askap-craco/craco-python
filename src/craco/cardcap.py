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
import ast

log = logging.getLogger(__name__)
hostname = socket.gethostname()

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

FINE_CHANBW = 1.0*32./27./64. # MHz
FINE_TSAMP = 1.0/FINE_CHANBW # Microseconds
NFPGA = 6 # number of FPGAs per card
NCHAN = 4 # number of CRACO output channels per FPGA

def uint8tostr(d):
    if isinstance(d, str):
        s = d
    else:
        s = ''.join(map(chr, d))

    return s.strip().replace('\x00','')


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

class CardcapFile:
    def __init__(self, fname, workaround_craco63=False):
        self.fname = fname
        hdr1  = fits.getheader(fname)
        mainhdr = fits.getheader(fname, 1)
        hdr_nbytes = len(str(hdr1)) + len(str(mainhdr))
        self.nbl = mainhdr.get('NBL', 465)
        self.debughdr = int(mainhdr.get('DEBUGHDR')) == 1
        self.polsum = int(mainhdr.get('POLSUM')) == 1
        self.dtype = get_single_packet_dtype(self.nbl, self.debughdr, self.polsum)

        self.hdr1 = hdr1
        self.mainhdr = mainhdr
        self.hdr_nbytes = hdr_nbytes
        self.workaround_craco63 = workaround_craco63
        self.pkt0 = self.load_packets(count=1) # load inital packet to get bat and frame IDx

    @property
    def indexes(self):
        '''
        Returns all indexes as a 4-typle
        return (products, revproducts, auto_products, cross_products)
        '''
        return get_indexes(self.nant)


    @property
    def nant(self):
        '''
        Returns number of antennas in this file
        '''
        return self.mainhdr['NANT']


    @property
    def card_frequencies(self):
        '''
        Returns a numpy array of channel frequencies for all FPGAs and channels in this card
        
        :returns: np array with shape (NFPGA, NCHAN) with values in MHz
        '''
        fch1 = self.mainhdr['FCH1']
        foff_chan = self.mainhdr['FOFFCHAN']
        foff_fpga = self.mainhdr['FOFFFPGA']
        f = np.zeros((NFPGA, NCHAN))
        for fpga in range(NFPGA):
            for chan in range(NCHAN):
                f[fpga, chan] = fch1 + foff_chan*chan + foff_fpga*fpga

        return f

    @property
    def syncbat(self):
        '''
        Returns sync bat as an int
        '''
        return int(self.mainhdr['SYNCBAT'], 16)

    @property
    def frequencies(self):
        '''
        Returns a numpy array of channel frequencies for only the FPGAs and channels in this file
        :returns: numpy array of shape [len(self.fpgas), NCHAN]
        '''
        this_file_fpgas = self.fpgas
        this_file_freqs = self.card_frequencies[this_file_fpgas - 1, :]
        return this_file_freqs

    @property
    def nbeam(self):
        '''
        Number of beams in this file
        '''
        hnbeam = self.mainhdr['BEAM']
        if hnbeam == -1:
            nbeam = 36
        else:
            nbeam = 1
        return nbeam

    @property
    def beams(self):
        '''
        Returns a numpy array containing the list of beams in this file
        Zero indexed
        '''
        if self.nbeam == 36:
            b = np.arange(36, dtype=np.int)
        else:
            thebeam = self.mainhdr['BEAM']
            b = np.array([thebeam], dtype=np.int)

        return b

    @property
    def npol(self):
        npol = 1 if self.polsum else 2
        return npol

    @property
    def nint_per_packet(self):
        '''
        if polsum is enabled, we get 2 integrations per set of debug headers
        '''
        
        if self.npol == 2:
            return 1
        else:
            return 2

    @property
    def fpgas(self):
        '''Returns a list of fpgas in this file (1 based)'''
        fstr = self.mainhdr['FPGA']
        fpga = ast.literal_eval(fstr)
        return np.array(fpga, dtype=np.int)


    @property
    def bat0(self):
        '''
        Returns first BAT in the file
        '''
        return self.pkt0['bat'][0]

    @property
    def frame_id0(self):
        '''
        Returns first frameid in teh file
        '''
        return self.pkt0['frame_id'][0]

    @property
    def mjd0(self):
        '''
        Returns astropy time of the first frame in the file

        :returns: Astropy Time Formatted as an mjd with scale=tai
        '''
        return self.time_of_frame_id(self.frame_id0)

    @property
    def isempty(self):
        '''
        Returns true if this file is empty
        As a bug, NAXIS2=1 is also empty
        '''
        return len(self) <= 1

    def __len__(self):
        '''
        Returns the value of NAXIS2 in the header - which should (hopefully) 
        align with the number of packets captured in this file
        Although - if its' empty, NAXIS2 will be 1
        Even though it's empty. That is a bug.
        '''
        return self.mainhdr['NAXIS2']

    def time_of_frame_id(self, frame_id):
        '''
        Returns an astropy time of the given frame ID

        :returns: Astropy Time Formatted as an mjd with scale=tai
        '''
        bat_of_frame_id = self.syncbat + int(frame_id)*27*2 # actually its frame_id * 27/32 * 64
        frame_id_t = Time(bat_of_frame_id / 1e6 / 3600 / 24, format='mjd', scale='tai')
        return frame_id_t
        

    def packet_iter(self, npackets=1):
        '''
        Returns an interator that interates through the packets in blocks of npackest
        :npackets: number of packets per block
        '''

        # workaroudn a bug - if it's empty the NAXIS2=1 and there's a single data value in there
        # just detect this condition and skip
        if len(self) > 1:
            with open(self.fname) as f:
                f.seek(self.hdr_nbytes)

                while True:
                    packets = np.fromfile(f, dtype=self.dtype, count=npackets)
                    if len(packets) != npackets:
                        break
                
                    if self.workaround_craco63:
                        shift = -1
                        packets['data'] = np.roll(packets['data'], shift, axis=0)

                    log.debug('yielding packets %s %s', packets.shape, packets['data'].shape)

                    yield packets
    
    def load_packets(self, count=-1, pktoffset=0):
        if len(self) == 1:# Work around bug 
            return np.array([], dtype=self.dtype)
        
        with open(self.fname) as f:
            f.seek(self.hdr_nbytes + self.dtype.itemsize*pktoffset)
            packets = np.fromfile(f, dtype=self.dtype, count=count)
            if self.workaround_craco63:
                shift = -1
                packets['data'] = np.roll(packets['data'], shift, axis=0)
        
        return packets

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

    if su m_pols is True, npol=1, nint=2
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
        log.info(f"Requests issued Enqueued={rx.numWorkRequestsEnqueued} missing={rx.numWorkRequestsMissing} total={rx.numTotalMessages} qloading={rx.currentQueueLoading} min={rx.minWorkRequestEnqueue} region={rx.regionIndex}")

    def write_data(self, w):
        rx = self.rx
        msg_size = self.ccap.msg_size
        rx.waitRequestsCompletion()
        rx.pollRequests()
        ncompletions = rx.get_numCompletionsFound()
        #nmissing = rx.get_numMissingFound()
        completions = rx.get_workCompletions()
        self.curr_packet_time = time.process_time()
        if self.total_completions == 0 and ncompletions > 0:
            self.first_packet_time = self.curr_packet_time

        self.total_completions += ncompletions
        assert ncompletions == len(completions)
        num_cmsgs = self.ccap.num_cmsgs
        #print(f'\rCompletions={ncompletions} {len(completions)}')
        beam = self.ccap.values.beam
        if ncompletions > self.max_ncompletions:
            log.critical(f'{self.values.block}/{self.values.card}/{self.fpga} increased completions {self.max_ncompletions}->{ncompletions}')
            self.max_ncompletions = ncompletions
        
        for c in completions:
            #assert c.status == ibv_wc_status.IBV_WC_SUCCESS
            index = c.wr_id
            nbytes = c.byte_len
            assert nbytes == msg_size, f'Unexpected messages size. Was={nbytes} expected={msg_size}'
            immediate = c.imm_data
            #immediate = socket.ntohl(immediate)
            self.total_bytes += nbytes

            immdiff = 2048 # fixed samples per frame

            if self.curr_imm is None:
                expected_immediate = immediate
            else:
                expected_immediate = (self.curr_imm + immdiff) % (1<<32)

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
                
            if immediate != expected_immediate:
                self.total_missing += diff
                log.critical(f'{hostname} {self.values.block}/{self.values.card}/{self.fpga} MISSED PACKET imm={immediate}={hex(immediate)} fid={fid}={hex(fid)} fid_diff={fid_diff}expected={expected_immediate} Diff={diff}  nmiss={self.total_missing} nbytes={nbytes} qloading={rx.currentQueueLoading}')
            

            self.curr_imm = immediate
            self.curr_fid = fid

            if self.ccap.values.workaround_craco63:
                d['data'] = np.roll(d['data'], shift=-1, axis=0)

            if self.ccap.values.tscrunch != 1:
                # OK this is slow, but it works
                dout = np.empty(d.shape[0], dtype=d.dtype)
                for field in ('frame_id','bat','beam_number','sample_number','channel_number','fpga_id','nprod','flags','zero1','version','zero3'):
                    dout[field] = d[field][:,0]

                dout['data'] = d['data'].mean(axis=1, dtype=np.float32).astype(np.int16)
                d = dout

            if beam is not None:
                mask = d['beam_number'] == beam 
                d = d[mask]


            if w is not None:
                w.write(d) # write to fitsfield
            
        rx.issueRequests()

    def __del__(self):
        log.info(f'Deleting RX for card {self.ccap.values.card}  FPGA {self.fpga}')
        del self.rx
        del self.rdma_buffers

class CardCapturer:
    def __init__(self, values, primary=False):
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

        if values.pol_sum: # if polsum is enabled, we get 2 integrations per set of debug headers
            nint_per_packet = 2
        else:
            nint_per_packet = 1

        nintpacket_per_frame = nint_per_frame // nint_per_packet # number of integrations per frame
        npacket_per_msg= nbeam*nchan*nintpacket_per_frame
            
        enable_debug_header = values.debug_header
        packet_dtype = get_single_packet_dtype(nbl, enable_debug_header, values.pol_sum)
        self.packet_dtype = packet_dtype
        msg_size = packet_dtype.itemsize*npacket_per_msg
        num_blks = values.num_blks
        num_cmsgs = values.num_cmsgs

        self.msg_size = msg_size
        self.npacket_per_msg = npacket_per_msg
        self.nbeam = nbeam
        self.nchan = nchan
        self.nint_per_frame = nint_per_frame
        self.nint_per_packet = nint_per_packet
        self.nintpacket_per_frame = nintpacket_per_frame
        self.nintout_per_frame = nintpacket_per_frame // self.values.tscrunch

        assert self.values.tscrunch == 1 or self.values.tscrunch * self.values.samples_per_integration == 2048, 'Invalid tscrunch - it must be 1 or multiply SPI to 2048'
        self.msg_shape = (num_cmsgs, self.nbeam*self.nchan, self.nintpacket_per_frame)
        self.out_shape = (self.nbeam*self.nchan, self.nintout_per_frame)
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
        hdr['NBEAM'] = (self.nbeam, 'Number of beams being downlaoded (always 36)')
        hdr['NCHAN'] = (self.nchan, 'Number of channels per FPGA (4)')
        hdr['NTPFM'] = (self.nint_per_frame, 'Total number of integrations per frame')
        hdr['NTPKFM'] = (self.nintpacket_per_frame, 'Total number of packet integrations per frame')
        hdr['MSGSHAPE'] = (str(self.msg_shape), 'Shape of a message buffer')
        hdr['TSCRUNCH'] = (self.values.tscrunch, 'Tscrunch factor')
        hdr['OUTSHAPE'] = (str(self.out_shape), 'Shape of file output')
        hdr['NTOUTPFM'] = (self.nintout_per_frame, 'Number of output integraitons per frame, after tscrunch')
        
        self.hdr = hdr
        if values.prefix != 'ma':
            self.pvhdr('md2:targetName_O', 'TARGET','Target name from metadata')
            self.pvhdr('md2:scanId_O', 'SCANID','Scan ID from metadata')
            self.pvhdr('md2:schedulingblockId_O', 'SBID','SBID rom metadata')
            self.pvhdr('F_options:altFirmwareDir_O', 'FWDIR', 'Alternate firmware directory')

        log.info(f'Shelf {shelf} card {card} Receiving data from {len(values.fpga)} fpgas: {values.fpga}')
        self.configure_args  = (fpgaMask, enMultiDest, enPktzrDbugHdr, enPktzrTestData, lsbPosition, sumPols, integSelect)

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
            self.stop(wait=False)
            self.finish_time = time.process_time()

            self.report_stats()
            if self.fitsout is not None:
                self.fitsout.close()
                
            self.fitsout = None
            #ctrl.write(f'acx:s{shelf:02d}:evtf:craco:enable', 0, wait=False)
                

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
        for f in self.fpga_cap:
            del f
        del self.fpga_cap
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

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Raw download and saving ROCE data from correlator cards', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d','--device', help='RDMA device', default='mlx5_0')
    parser.add_argument('-p','--port', help='RDMA port', type=int, default=1)
    parser.add_argument('-g','--gid-index', help='RDMA GID index', type=int, default=2)
    parser.add_argument('-n','--num-blks', help='Number of ringbuffer slots', type=int, default=16)
    parser.add_argument('-c','--num-cmsgs', help='Numebr of messages per slot', type=int, default=1)
    parser.add_argument('--num-msgs', help='Total number of messages to download before quitting', default=-1, type=int)
    parser.add_argument('-e','--debug-header', help='Enable debug header', action='store_true', default=False)
    parser.add_argument('--prompt', help='Prompt for PSN/QPN/GID from e.g. rdma-data-transport/recieve -s', action='store_true', default=False)
    parser.add_argument('-f', '--outfile', help='Data output file')
    parser.add_argument('-b','--block',help='Correlator block to talk to', default=7, type=strrange) 
    parser.add_argument('-a','--card', help='Card range to talk to', default=1, type=strrange)
    parser.add_argument('-k','--fpga', help='FPGA range to talk to', default='1-6', type=strrange)
    parser.add_argument('--prefix', help='EPICS Prefix ma or ak', default='ma')
    parser.add_argument('--enable-test-data', help='Enable test data mode on FPGA', action='store_true', default=False)
    parser.add_argument('--beam', default=None, type=int, help='Which beam to save (default=all)')
    parser.add_argument('--lsb-position', help='Set LSB position in CRACO quantiser (0-11)', type=int, default=11)
    parser.add_argument('--samples-per-integration', help='Number of samples per integration', type=int, choices=(16, 32, 64), default=32)
    parser.add_argument('--tscrunch', help='Tscrunch by this factor before saving', type=int, default=1, choices=(1,2,4,8,16,32,64))
    parser.add_argument('--mpi', action='store_true', help='RunMPI version', default=False)
    parser.add_argument('--workaround-craco63', action='store_true', help='CRACO63 workaround', default=False)

    pol_group = parser.add_mutually_exclusive_group(required=True)
    pol_group.add_argument('--pol-sum', help='Sum pol mode', action='store_true')
    pol_group.add_argument('--dual-pol', help='Dual pol mode', action='store_true')
    
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    if values.mpi:
        import mpi4py.rc
        mpi4py.rc.threads = False
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numprocs = comm.Get_size()

        block_cards  = []
        procid = 0
        for blk in values.block:
            for crd in values.card:
                for fpga in values.fpga:
                    block_cards.append((blk, crd, fpga))
                    procid += 1
                    if procid > numprocs:
                        break

        primary = rank == 0

        # Before we start, we need to stop everything so nothign gets confused
        if rank == 0:
            ctrl = CracoEpics(values.prefix+':')
            ctrl.stop()
            
        comm.Barrier()

        if rank < len(block_cards):
            my_block, my_card, my_fpga =  block_cards[rank]
            my_values = copy.deepcopy(values)
            my_values.card = my_card
            my_values.block = my_block
            my_values.fpga = [my_fpga]

            devices = ['mlx5_0', 'mlx5_1']
            #devices =['mlx5_0','mlx5_0']
            devidx = my_fpga % 2
            my_values.device = devices[devidx]


            if values.outfile is None:
                my_values.outfile = None
            else:
                my_values.outfile = values.outfile.replace('.fits','')
                my_values.outfile += f'_b{my_block:02d}_c{my_card:02d}+f{my_fpga:d}.fits'
                
            log.info(f'MPI CARDCAP: My rank is {rank}/{numprocs}. Downloaing card={my_values.card} block {my_values.block} fpga={my_values.fpga} to {my_values.outfile}')
            ccap = CardCapturer(my_values, primary)
        else:
            log.info(f'Launched too may processes for me to do anything useful. Rank {rank} goin to sleep to get out of the way')
            my_values = None
            ccap = None

        
        if rank == 0:
            ccap.configure()
            # disable all cards, and enable only the ones we want
            blk = values.block[0]
            assert len(values.block) == 1, 'Cant start like that currently'

            # Enable only the cards we want.
            ctrl.enable_card_events(blk, values.card)
            # Normally do start() here but it would re-enable everything,
            # so just start this block
            #ctrl.start_block(blk)
            ccap.start()

        comm.Barrier()

        if rank < len(block_cards):
            ccap.do_writing()

        comm.Barrier()
        
        if rank == 0:
            ccap.stop()

        ncomplete = -1 if ccap is None else ccap.total_completions[0]
        completions = comm.gather(ncomplete)

        nmissing = -1 if ccap is None else ccap.total_missing[0]
        nmissing = comm.gather(nmissing)

        if rank == 0:
            log.info('Total completions=%s missing=%s', completions[:len(block_cards)], nmissing[:len(block_cards)])
            
    else: # not mpi
        primary = True
        log.exception('MPI setup failed. Falling back to vanilla goodness of wonderment')
        my_values = values
        my_values.card = values.card[0]
        my_values.block = values.block[0]
        ccap = CardCapturer(my_values, primary)
        ccap.stop()
        ccap.configure()
        ccap.start()
        ccap.do_writing()



if __name__ == '__main__':
    _main()
