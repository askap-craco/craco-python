#!/usr/bin/env python
"""
Class to merge cardcap files

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco.cardcapfile import CardcapFile, NCHAN,NSAMP_PER_FRAME, NBEAM
from numba.types import List
from craco.utils import get_target_beam
from typing import List

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def none_next(i):
    try:
        b = next(i)
    except StopIteration:
        b = None

    return b

def frame_id_iter(i, fid0):
    '''
    Generator that yields blocks of data
    It will yield a block with the given frame id startign at fid0 and incrementing
    by fidoff
    if no block has the given frame ID, it will yield None
    '''

    frame_id = fid0
    fidoff = np.uint64(NSAMP_PER_FRAME)
    assert isinstance(fid0, np.uint64), f'FID0 is the wrong type {fid0} {type(fid0)}'
    currblock = None
    last_frameid = frame_id
    last_bat = 0
    while True:
        if currblock is None or currblock['frame_id'].flat[0] < frame_id:
            try:
                currblock = next(i)
            except StopIteration:
                break

        curr_bat = currblock['bat'][0]
        curr_frameid = currblock['frame_id'][0]

        #assert curr_frameid >= frame_id, f'Block should have a timestamp now or in the future. curr_frameid={curr_frameid} frame_id={frame_id}'

        if curr_frameid == frame_id:
            b = currblock
            log.debug(f'HIT frame_id={frame_id} hit {curr_bat}')
        else:
            log.debug(f'MISS expected frame_id={frame_id} current={curr_frameid} fidoffset ={fidoff} last_frameid={last_frameid} curr-last={int(curr_frameid) - int(last_frameid)} expected-curr={frame_id-curr_frameid} BAT curr-last={curr_bat - last_bat}')
            b = None

        if curr_frameid >= fid0:
            assert b is None or (curr_frameid == frame_id), f'Logic error. Block should be none or Frame IDs should be equal. curr_frameid={curr_frameid} frameid={frame_id} block is None?{b is None}'
            
            yield frame_id, b
            frame_id += fidoff
            
        last_frameid = curr_frameid
        last_bat = curr_bat


class CcapMerger:
    def __init__(self, fnames, headers=None):
        ''' Need to give it file names or headers
        '''
        self.ccap = []
        self.fnames = fnames
        if self.fnames is not None:
            for f in self.fnames:
                try:
                    cc = CardcapFile(f)
                    self.ccap.append(cc)
                except:
                    log.exception('Error opening file %s', f)
                    raise
        else:
            assert headers is not None
            assert len(headers) > 0, 'Empty list for headers'
            assert len(headers[0]) > 0, 'Empty header'
            self.ccap = [CardcapFile.from_header_string(hdr) for hdr in headers]

        nfpga = len(self.ccap[0].fpgas)
        nfiles = len(self.ccap)
        all_freqs = np.zeros((nfiles, nfpga, NCHAN))

        for ic, c in enumerate(self.ccap):
            assert len(c.fpgas) == nfpga, 'Differing numbers of fpgas in the files'
            all_freqs[ic, :, :] = c.frequencies

        self.all_freqs = all_freqs
            
        fidxs = np.argsort(all_freqs.flat).reshape(nfiles, nfpga, NCHAN)
        self.fidxs = fidxs
        self.tscrunch = self.ccap[0].tscrunch
        self.nbeam = self.ccap[0].nbeam
        self.nbl = self.ccap[0].mainhdr['NBL']
        self.__npol = self.ccap[0].npol
        self.all_freqs = all_freqs
        self.nchan_per_file = nfpga*NCHAN

    @classmethod
    def from_headers(cls, headers: List[str]):
        '''
        Creates a merger file from a list of headers
        Some properties won't be available because they need to see the data
        '''
        return CcapMerger(None, headers)

    @property
    def frame_ids(self):
        fids  = [c.frame_id0  if not c.isempty else None for c in self.ccap]
        return fids

    @property
    def frame_id0(self):
        return min([fid for fid in self.frame_ids if fid is not None])

    @property
    def bats(self):
        b = [c.bat0 if not c.isempty else None for c in self.ccap]
        return b


    @property
    def beams(self):
        '''
        Returns array of beams available in thei file
        '''
        return self.ccap[0].beams

    @property
    def fcent(self):
        '''
        Returns center frequency - units of MHz
        '''
        return self.all_freqs.mean()

    @property
    def fch1(self):
        '''
        Returns first channel frequency
        '''
        return self.all_freqs.flat[0]

    @property
    def foff(self):
        '''
        Returns channel interval - MHz
        '''
        fsort = np.array(sorted(self.all_freqs.flat))
        fdiff = fsort[1:] - fsort[:-1]
        foff = fdiff.mean()
        return foff

    @property
    def nchan(self):
        return self.all_freqs.size

    @property
    def npol(self):
        return self.__npol

    @property
    def mjd0(self):
        t = self.ccap[0].time_of_frame_id(self.frame_id0)
        log.debug(f'MJD0 for {self.ccap[0].fname} is {t}')
        return t

    @property
    def antnos(self):
        '''
        Returns a list 1-based of antenna numbers that are used in this file

        By default (and currently) it returns 1-nant
        '''
        antnos = [a+1 for a in range(self.nant)]
        return antnos
        
    @property
    def nant(self):
        n = self.ccap[0].mainhdr['NANT']
        return n

    @property
    def npol(self):
        return self.ccap[0].npol

    @property
    def indexes(self):
        return self.ccap[0].indexes

    @property
    def inttime(self):
        t = self.ccap[0].mainhdr['TSAMP']*self.ccap[0].tscrunch
        return t

    @property
    def nint_per_frame(self):
        return self.gethdr('NTPFM')

    @property
    def ntpkt_per_frame(self):
        return self.ccap[0].ntpkt_per_frame

    @property
    def nint_per_packet(self):
        return self.ccap[0].nint_per_packet

    @property
    def nt_per_frame(self):
        return self.ccap[0].nt_per_frame

    @property
    def tscrunch_bug(self):
        return self.ccap[0].tscrunch_bug

    @property
    def npackets_per_frame(self):
        return self.ccap[0].npackets_per_frame

    @property
    def dtype(self):
        return self.ccap[0].dtype

    @property
    def target(self):
        return self.gethdr('TARGET')

    def gethdr(self, key):
        return self.ccap[0].mainhdr[key]
    
    def fid_to_mjd(self, fid):
        return self.ccap[0].time_of_frame_id(fid)

    def packet_iter(self, frac_finished_threshold=0.9, beam=None, start_fid=None):
        '''
        Returns an iterator that returns arrays of blocks of data but without converting them 
        into a masked array
        Yields a tuple containing ((fid, packets,) fids)
        Packets are the packets from each input
        fids is a tuple of frame_ids, one from each input
        :start_fid:  starting frame ID to get data for
        '''
        if start_fid is None:
            start_fid = self.frame_id0
            
        iters = [frame_id_iter(c.frame_iter(beam), start_fid) for c in self.ccap]
        #packets = List() # TODO: Make NUMBA happy with List rather than  array

        assert 0 < frac_finished_threshold <= 1, f'Invalid fract finished threshoold {frac_finished_threshold}'
        
        while True:
            packets = []
            finished_array = []
            for iterno, i in enumerate(iters):
                try:
                    packets.append(next(i))
                    finished_array.append(False)
                except StopIteration:
                    packets.append((None,None))
                    finished_array.append(True)

            assert len(packets) == len(iters)
            flagged_array = [p is None or p[1] is None for p in packets]
            fids = [None if p is None else p[0] for p in packets]
            num_finished = sum(finished_array) # True is 1 and Flase is 0, so this is the number of finished things
            frac_finished = num_finished/len(finished_array)
            finished = frac_finished >= frac_finished_threshold

            log.debug('Got packets. Finished=%s  %s %s - flagged %s FIDS=%s', finished, frac_finished, finished_array, flagged_array, fids)
            
            if finished:
                break

            yield packets, fids

    def block_iter(self, frac_finished_threshold=0.9, beam=None):
        '''
        Returns an iterator that returns blocks of data
        Blocks have shape (nchan,nbeam,ntime,nbl,npol,2), dtype=np.int16 and are masked arrays
        Mask is true (invalid) if frameID missing from file, or file has terminated

        Iteration finishes when the fraction of files that have finished is greater than frac_finihsed_threshold
        :beam: choose beam. None means whatever is in the source. -1 means force all or error. else choose a beam number
        '''
        for packets, fids in self.packet_iter(frac_finished_threshold, beam):
            outfid, dout = self.merge_and_mask_packets(packets, beam)
            yield outfid, dout

    def merge_and_mask_packets(self, packets, beam=None):
        nfile = len(self.ccap)
        assert self.nchan == len(self.ccap)*self.nchan_per_file
        nint_total = self.ntpkt_per_frame*self.nint_per_packet
        nbeam = NBEAM if beam is None or beam == -1 else 1

        shape = (nfile, self.nchan_per_file, nbeam, self.ntpkt_per_frame, self.nint_per_packet, self.nbl, self.npol, 2)
        dout = np.zeros(shape, dtype=np.int16)
        mask = np.zeros(shape, dtype=np.bool) # default is False which means not masked - i.e is valid
        newshape = (self.nchan, nbeam, nint_total, self.nbl, self.npol, 2)
        log.debug('Initial dout shape=%s final shape=%s beam=%s nbeam=%s', shape, newshape, beam, nbeam)
        outfid = None
        
        for ip, (fid, p) in enumerate(packets):
            assert self.fidxs.shape[1] == 1, 'Can only handle single FPGA files'
            #log.debug('ip=%s fid=%s p.shape=%s dout.shape=%s', ip, fid, p.shape, dout.shape)
            
            if p is None:
                log.debug(f'Flagged {self.ccap[ip].fname} fid={fid}')
                mask[ip,...] = True
                # data is already 0, but now it's masked anyway
            else: # mask is already false = valid data
                outfid = fid
                if nbeam == 1: # Data order is just (4, 1, nint_per_frame)
                    p.shape = (NCHAN, 1, self.ntpkt_per_frame)

                    if self.tscrunch_bug:
                        dblk = p['data'].mean(axis=3, keepdims=True)
                        dout[ip,:] = dblk
                    else:
                        dout[ip,:] = p['data']
                else:
                    # This reshapes for teh beams 0-31 first, then beams 32-35 next
                    assert nbeam == 36
                    p.shape = (NCHAN*nbeam, self.ntpkt_per_frame)
                                    
                    blk1 = p[:32*4, :] 
                    blk1.shape = (NCHAN,32, self.ntpkt_per_frame) # first 32 beams
                    blk2 = p[32*4:]
                    blk2.shape = (NCHAN,4, self.ntpkt_per_frame) # final 4 beams
                    if self.tscrunch_bug: # average over 2 integrations
                        blk1d = blk1['data'].mean(axis=3, keepdims=True)
                        blk2d = blk2['data'].mean(axis=3, keepdims=True)
                    else:
                        blk1d = blk1['data']
                        blk2d = blk2['data']

                    dout[ip,:, :32, ...] = blk1d
                    dout[ip,:, 32:, ...] = blk2d

        dout.shape = newshape
        mask.shape = newshape
        
        # permute frequency channels
        dout = dout[self.fidxs.flatten(),...]
        mask = mask[self.fidxs.flatten(),...]
        dout = np.ma.masked_array(dout,mask)
                
        return (outfid, dout)

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-b','--beam', type=int, help='Choose beam to dump')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    merger = CcapMerger(values.files)
    print('FID shape mean std max')
    for fid, blk in merger.block_iter(beam=values.beam):
        d = blk
        print(fid, blk.shape, d.mean(), d.std(), d.max())

if __name__ == '__main__':
    _main()
