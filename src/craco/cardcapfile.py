#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
import warnings
from astropy.time import Time
from astropy.io import fits
from  astropy.io.fits.header import Header
from craco.utils import beamchan2ibc
import ast

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

FINE_CHANBW = 1.0*32./27./64. # MHz
FINE_TSAMP = 1.0/FINE_CHANBW # Microseconds
NFPGA = 6 # number of FPGAs per card
NCHAN = 4 # number of CRACO output channels per FPGA
NBEAM = 36 # number of beams per FPGA (or even per ASKAP!)
NSAMP_PER_FRAME = 2048 # Number of fine filterbank samples per beamformer frame

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

def get_single_packet_dtype(nbl: int, enable_debug_hdr: bool, sum_pols: bool=False, override_nint: int=None):
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

    if override_nint is not None:
        nint = override_nint
                     
    dtype.append(('data', '<i2', (nint, nbl, npol, 2)))
    
    return np.dtype(dtype)

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
    def __init__(self, fname, mainhdr=None, workaround_craco63=False):
        '''
        Supply the file name, or else mainhdr as a pyfits Header
        If you don't supply a file, some values won't work
        '''
        self.fname = fname
        if fname is not None:
            hdr1  = fits.getheader(fname)
            mainhdr = fits.getheader(fname, 1)
        else:
            assert mainhdr is not None
            hdr1 = Header()

        self.hdr1 = hdr1
        hdr_nbytes = len(str(hdr1)) + len(str(mainhdr))
        self.nbl = mainhdr.get('NBL', 465)
        self.debughdr = int(mainhdr.get('DEBUGHDR')) == 1
        self.polsum = int(mainhdr.get('POLSUM')) == 1
        self.dtype = get_single_packet_dtype(self.nbl, self.debughdr, self.polsum)
        self.mainhdr = mainhdr
        self.hdr_nbytes = hdr_nbytes
        self.workaround_craco63 = workaround_craco63
        if self.mainhdr.get('NTOUTPFM', None) == 0:
            warnings.warn(f'File {fname} has tscrunch/polsum bug. Will tscrunch final integrations')
            self.tscrunch_bug = True
        else:
            self.tscrunch_bug = False

        if fname is None:
            self.pkt0 = None
        else:
            self.pkt0 = self.load_packets(count=1) # load inital packet to get bat and frame IDx

    @classmethod
    def from_header_string(cls, hdrstring:str):
        '''
        Creates a cardcap file from the given header string
        maindhr is set as a FITS header.
        Use this to get a hold of various properties in the file
        e.g. nant, indexes etc
        If the file doesn't exist on  your system, then reading data won't work 
        and you won't be able to get start times etc.
        '''
        mainhdr = Header.fromstring(hdrstring)
        return CardcapFile(None, mainhdr)
        
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
    def target(self):
        return self.mainhdr['TARGET']

    @property
    def sbid(self):
        return self.mainhdr['SBID']

    @property
    def scanid(self):
        return self.mainhdr['SCANID']

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
        if self.nbeam == NBEAM:
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
            nint = 1
        else:
            if self.tscrunch_bug: # We'll scrunch them together
                nint = 1
            else:
                nint = 2

        return nint
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
        return len(self) == 0

    @property
    def tscrunch(self):
        return self.mainhdr.get('TSCRUNCH', 1)

    @property
    def ntpkt_per_frame(self):
        if self.tscrunch_bug:
            ntpkt = 1
        else:
            ntpkt = self.mainhdr['NTPKFM'] // self.tscrunch

        return ntpkt

    @property
    def npackets_per_frame(self):
        '''
        Number of packets per beamformer frame (110ms)
        is NCHAN * the numebr of beams recorded * number of time packets
        Don't forget: If npol=1 then you get 2 integrations per packet
        '''
        npkt = NCHAN*self.nbeam*self.ntpkt_per_frame
        return npkt


    def __len__(self):
        '''
        Returns the value of NAXIS2 in the header - which should (hopefully) 
        align with the number of packets captured in this file
        Although - if its' empty, NAXIS2 will be 1
        Even though it's empty. That is a bug.
        '''
        nax2 = self.mainhdr['NAXIS2']
        if nax2 == 1: # Bug - file not properly close and its empty
            nbytes = os.path.getsize(self.fname)
            datalen = nbytes - self.hdr_nbytes
            groupsize = self.dtype.itemsize
            ngroups = datalen // groupsize
            if ngroups == 1: # bug - if it wasn't closed properly and no data arrived, it has size=1. Just set it to zer
                mylen = 0
            else:
                mylen = ngroups
                
            warnings.warn(f'CCAP file {self.fname} was not closed correctly. Estimating ngroups from size={nbytes} datalen={datalen} len={ngroups}')
        else:
            mylen = nax2

        return mylen

    def time_of_frame_id(self, frame_id):
        '''
        Returns an astropy time of the given frame ID

        :returns: Astropy Time Formatted as an mjd with scale=tai
        '''
        bat_of_frame_id = self.syncbat + int(frame_id)*27*2 # actually its frame_id * 27/32 * 64
        frame_id_t = Time(bat_of_frame_id / 1e6 / 3600 / 24, format='mjd', scale='tai')
        return frame_id_t

    def __fix__(self, packets):
        if self.workaround_craco63:
            shift = -1
            packets['data'] = np.roll(packets['data'], shift, axis=0)

        if self.tscrunch_bug:
            packets['data'] = packets['data'].mean(axis=1, keepdims=True) 

        return packets


    def frame_iter(self, beam=None):
        '''
        Returns an iterator over a frame's worth of data. The frame is a 110ms Beamformer Frame.
        No matter how many beams, channels, integraitons polsummed etc that have happend, you get exactly one frame's worth.
        Optionally let's you specify a particular beam in which case you only get that beam
        NCHAN is always 4. Don't be crazy and think that you'd like to filter on channel. That's crazy because the channels between FPGAs are all 
        interleaved. Crazy person.

        :beam: If None, then it just gives you whatever beams are in the file without checking
        If specified (0--35) then it will give you only the beam you asked for
        if -1 then we're specfically asking for "all 36 beams". If the file doesn't contain all beams, then it throws ValueError
        Can't ask for anything other than 1 or 36 beams. It explodes my brain.

        :returns: np array of dtype=self.dtype = length will be NCHAN*nbeams*self.ntpkt_per_frame where
        nbeams is 36 if beam==-1, 1 or 36 if beam is None (dependent on the file contents) and 1 if beam is specified.
        If nbeam=1 it'll just be length=NCHAN*ntpkt_per_frame with the chape ovbious. If nbeam=36 it'll be in cracy beamformer ibc order
        which is nuts.
        :see: ibc2beamchan to convert ibc order to beam/channel indexes
        '''
        if beam is not None:
            if beam == -1 and self.nbeam != 36:
                raise ValueError(f'Requested all beams with beam=-1 but we only have {self.nbeam} beams={self.beams}')

            if beam not in self.beams:
                raise ValueError(f'Requested beam {beam} is not in cardcap file. Which does contain beams={self.beams}')

        packets_per_frame = self.npackets_per_frame
        packet_size_bytes = self.dtype.itemsize

        if beam is None or beam == -1:
            nbeam_out = NBEAM
        else:
            nbeam_out = 1


        if len(self) > 1: # work around bug
            with open(self.fname) as f:
                iframe = 0
                f.seek(self.hdr_nbytes)
                                        
                while True:
                    if beam is None or beam == -1:
                        packets = np.fromfile(f, dtype=self.dtype, count=packets_per_frame) # just reads sequentially
                        if len(packets) != packets_per_frame:
                            break

                        # do a little check
                        assert packets['beam_number'][0] == 0, f"Expected first beam to be zero. It was {packets['beam_number'][0]}"
                        # channel number isn't 0-4, its 0--large number, I thik
                        # assert packets['channel_number'][0] == 0, f"Expected first channel to be zero. It was {packets['channel_number'][0]}"
                        
                    else: # read 4 channels worth of ntpkts
                        packets = np.empty(NCHAN*self.ntpkt_per_frame, dtype=self.dtype) # 1 beam
                        for chan in range(NCHAN):
                            ibc = beamchan2ibc(beam, chan)
                            pkt_offset = iframe*packets_per_frame + ibc*self.ntpkt_per_frame
                            byte_offset = self.hdr_nbytes + pkt_offset*packet_size_bytes
                            f.seek(byte_offset)
                            n = self.ntpkt_per_frame
                            inpackets = np.fromfile(f, dtype=self.dtype, count=n)
                            if len(inpackets) != n:
                                break

                            assert inpackets['beam_number'][0] == beam, f"Expected first beam to be {beam}. It was {inpackets['beam_number'][0]}"
                            #assert packets['channel_number'][0] == chan, f"Expected first beam to be {chan}. It was {packets['channel_number'][0]}"
                            
                            # TODO: Work out how to read inplace rather than copying to improve performance
                            packets[chan*n:(chan+1)*n] = inpackets
                            

                    yield self.__fix__(packets)
                    iframe += 1
                    

    def packet_iter(self, npackets=1, beam=None):
        '''
        Returns an interator that interates through the packets in blocks of npackets
        :npackets: number of packets per block
        '''

        assert npackets > 0, 'Invalid npackets'

        # workaroudn a bug - if it's empty the NAXIS2=1 and there's a single data value in there
        # just detect this condition and skip
        if len(self) > 1:
            with open(self.fname) as f:
                f.seek(self.hdr_nbytes)

                while True:
                    packets = np.fromfile(f, dtype=self.dtype, count=npackets)
                    if len(packets) != npackets:
                        break
                
                    log.debug('yielding packets shape=%s data shape=%s, npackets=%s len(packets)=%s', packets.shape, packets['data'].shape, npackets, len(packets))

                    yield self.__fix__(packets)
    
    def load_packets(self, count=-1, pktoffset=0):
        if len(self) == 1:# Work around bug 
            return np.array([], dtype=self.dtype)
        
        with open(self.fname) as f:
            f.seek(self.hdr_nbytes + self.dtype.itemsize*pktoffset)
            packets = np.fromfile(f, dtype=self.dtype, count=count)
        
        return self.__fix__(packets)


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
