#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import glob
from craco.cardcap import NCHAN, NFPGA, NBEAM
from craco.cardcapfile import NSAMP_PER_FRAME
from craco import cardcap
from craco.cardcapmerger import CcapMerger, frame_id_iter
from craco.mpi_obsinfo import MpiObsInfo
import astropy.io.fits.header as header
from craco.mpi_appinfo import MpiPipelineInfo
from mpi4py import MPI

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class VisSource:
    pass


class CardCapFileSource:
    '''
    Gets data from a single card - i.e. 6 FPGAs
    '''
    def __init__(self, pipe_info:MpiPipelineInfo):
        self.pipe_info = pipe_info
        values = pipe_info.values
        cardidx = pipe_info.cardid
        icard = cardidx % len(values.card)
        iblk = cardidx  // len(values.card)
        card = values.card[icard]
        block = values.block[iblk]
        if values.cardcap_dir is not None:
            files = glob.glob(os.path.join(values.cardcap_dir, '*.fits'))
            log.info('Found %d cards in %s', len(files), values.cardcap_dir)
        else:
            files = values.files
            
        fstr = f'b{block:02d}_c{card:02d}'
        myfiles = sorted(filter(lambda f: fstr in f, files))
        log.info(f'Cardidx {cardidx} icard={icard} iblk={iblk} card={card} iblk={iblk} has {len(myfiles)} files ')
        assert len(myfiles) == NFPGA, f'Incorrect number of cards for cardidx={cardidx} {fstr} {myfiles} {files} in {values.cardcap_dir}'
        self.merger = CcapMerger(myfiles)
        self.myfiles = myfiles
        self.values = values
        self.card = card
        self.block = block

        # integration time comes from lots of different places and I probably don't do a great job of passing it around.
        # so I just want to check that the value in the arguments tallies with what's in the header
        spi = pipe_info.values.samples_per_integration
        assert spi * self.merger.nt_per_frame == NSAMP_PER_FRAME, \
            f'Invalid SPI: {spi} merger NT perframe={self.merger.nt_per_frame} expected product={NSAMP_PER_FRAME}'

    @property
    def fpga_headers(self):
        '''
        Returns the header for each file as a string
        '''
        hdrs = [str(ccap.mainhdr) for ccap in self.merger.ccap]
        return hdrs

    def start(self):

        '''
        If we have no frame ID then we return 0
        MPI.MAX reduce will ignore the 0 value
        '''
        
        if self.merger.is_all_empty:
            infid = 0
        else:
            infid = self.merger.frame_id0
            assert infid != 0
            
        infid = np.uint64(infid)
        fid0 = self.pipe_info.rx_comm.allreduce(infid, MPI.MAX)
        self.fid0 = np.uint64(fid0)
        return self.fid0

    def packet_iter(self, start_fid=None):
        assert self.fid0 is not None, 'Must call start()'
        return self.merger.packet_iter(start_fid=self.fid0, beam=self.values.beam)
    
class SyntheticVisSource:
    ''''
    Loads headers from cardcap files
    but returns synthetic data
    '''
    def __init__(self, pipe_info:MpiPipelineInfo, ncache=13):
        '''
        Instead of computing random stuff all the time, we just compute some random stuff once
        them loop through it.
        nchan prime is probably a good choice
        '''
        self._filesource = CardCapFileSource(pipe_info)
        self.merger = self._filesource.merger
        
        # Seed the RNG with the cardid
        cardid = pipe_info.cardid
        rng = np.random.default_rng(seed=cardid)
        
        self.packets = [[np.zeros((NBEAM*NCHAN, self.merger.ntpkt_per_frame), dtype=self.merger.dtype) for f in range(NFPGA)] for c in range(ncache)]
        for cache in self.packets:
            for pkt in cache:
                pkt['data'] = rng.standard_normal(pkt['data'].shape) * 1000

        self.__icache = 0
    
    @property
    def fpga_headers(self):
        return self._filesource.fpga_headers
    
    def start(self):
        return self._filesource.start()
    
    def packet_iter(self, nframes):
        fid = np.uint64(self._filesource.fid0)

        while True:            
            packets = self.packets[self.__icache]
            # make a copy of the packets so we can modify with injections
            packets = [p.copy() for p in packets]
            
            for d in packets:
                #d['data'] = (np.random.randn(*d['data'].shape)*50).astype(np.int16)
                d['bat'][0] = fid
                d['frame_id'][0][0] = fid

            yield (packets, [fid]*NFPGA)
            fid += NSAMP_PER_FRAME
            self.__icache = (self.__icache + 1) % len(self.packets)

    

class CardCapNetworkSource:
    def __init__(self, pipe_info:MpiPipelineInfo):
        block_cards  = []
        procid = 0
        cardno = 0
        values = pipe_info.values
        numprocs = pipe_info.rx_comm.Get_size()
        self.pipe_info = pipe_info
        self.skip_frames = 10*30 # skip this many this many 110ms beamformer frames before returning data. TODO: Get from cmdline.

        # assign all FPGAs to each rank
        for blk in values.block:
            for crd in values.card:
                cardno += 1
                block_cards.append((blk, crd, values.fpga))
                procid += 1
                if procid > numprocs:
                    break

        log.debug('numprocs %s block_cards %s', numprocs, block_cards)

        net_device = pipe_info.my_rank_info.net_dev
        self.ctrl = cardcap.MpiCardcapController(pipe_info.rx_comm,
                                                  pipe_info.values,
                                                  block_cards,
                                                  device=net_device)
        # make dummy merger so othe rpeople can get dtype and
        # other useful parameters
        self.merger = CcapMerger.from_headers(self.fpga_headers)
        this_block_card = block_cards[pipe_info.cardid]
        self.block, self.card = this_block_card[0:2]
        self.fid0 = None


    @property
    def fpga_headers(self):
        # convert to proper fits header. Insane.
        
        fitshdr = header.Header()
        for k,v in self.ctrl.ccap.hdr.items():
            fitshdr[k] = v

        return [str(fitshdr)]

    def start(self):
        # This will start a few seconds into the future. We'd better get our skates on
        start_bat = self.ctrl.configure_and_start()
        self.start_bat = start_bat # BAT for when CRACO Go event happens. Data starts on the next BF frame boundary (i.e. 2048 FIDs)
        self.init_fid0 = self.merger.get_fid0_from_start_bat(self.start_bat)
        self.fid0 = self.init_fid0 + np.uint64(self.skip_frames*NSAMP_PER_FRAME) # the frame ID iterator will skip this many frameids before starting
        log.info('Start bat was 0x%x=%d init_fid=%d skipping %d frames. new FID0=%d',
                self.start_bat, start_bat, self.init_fid0, self.skip_frames, self.fid0)
        
        return self.fid0

    def packet_iter(self, nframes):
        assert self.fid0 is not None, 'Havent called start'
        iters = [frame_id_iter(cap.packet_iterator(65536), self.init_fid0, nframes) for cap in self.ctrl.ccap.fpga_cap]
        # We need to keep looping over all FPGAs otehrwise we just get stuck on one.
        # so all FPGAs start at the original planned start bat.
        # we loop through and only yield when all FPGAs have an FID >= the origianllly specified fid0
        while True:
            try:
                fpga_data = [next(fiter) for fiter in iters]
                packets = [fd[1] for fd in fpga_data]
                fids = [fd[0] for fd in fpga_data]
                fid = fids[0]
                if fid >= self.fid0:
                    yield (packets, fids)
            except StopIteration:
                break

        self.ctrl.stop()
                

def open_source(pipe_info:MpiPipelineInfo):
    '''
    IF cardcap_dir is specified, it will open a cardcapfilesource 
    IF fake_cardcap_data is specified, it will open a syntheticvissource
    Otherwise, opens the network and goes the whole hog, you madman.
    
    - You go whole hog.
    - OOOOooh, what a burn!


    '''

    values = pipe_info.values
    if values.cardcap_dir is not None:
        if values.fake_cardcap_data is not None:
            src = SyntheticVisSource(pipe_info)
        else:
            src = CardCapFileSource(pipe_info)
    else:
        src = CardCapNetworkSource(pipe_info)

    return src
    

class VisBlock:
    '''
    Represents a block of data from a single source for a single beam.
    All baselines
    Block of NT integrations
    '''
    def __init__(self, data, iblk, info:MpiObsInfo, cas=None, ics=None):
        self._d = data
        self.iblk = iblk
        self.info = info
        self.cas = cas
        self.ics = ics

    @property
    def data(self):
        '''
        Returns visibilities
        '''
        return self._d

    @property
    def fid_start(self):
        '''
        Returns frame ID of the first sample in this block
        '''
        return self.info.fid_of_block(self.iblk)

    @property
    def fid_mid(self):
        '''
        returns frame ID for middle of this block
        '''
        # TODO: Check timestamp convention for for FID and mjd.
        # I think this is right
        # fid_start goes up by NSAMP_PER_FRAME = 2048 every block
        # We'll calculate the same UVWs for everything in this block. A bit lazy but not rediculous

        return self.fid_start + np.uint64(NSAMP_PER_FRAME//2)

    @property
    def mjd_mid(self):
        return self.info.fid_to_mjd(self.fid_mid)

    @property
    def uvw(self):
        return self.info.uvw_at_time(self.mjd_mid)

    @property
    def source_index(self):
        return self.info.source_index_at_time(self.mjd_mid)

    @property
    def antflags(self):
        return self.info.antflags_at_time(self.mjd_mid)

    @property
    def baseline_flags(self):
        '''
        Returns  a length nbl array of bool. True if either antenna is flagged
        '''
        af = self.antflags
        blflags = np.array([af[blinfo.ia1] | af[blinfo.ia2] for blinfo in self.info.baseline_iter()])
        return blflags

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
