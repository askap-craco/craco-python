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
from craco.cardcap import NCHAN, NFPGA
from craco.cardcapfile import NSAMP_PER_FRAME
from craco import cardcap
from craco.cardcapmerger import CcapMerger, frame_id_iter
import astropy.io.fits.header as header
from mpi4py import MPI

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class VisSource:
    pass


class CardCapFileSource:
    '''
    Gets data from a single card - i.e. 6 FPGAs
    '''
    def __init__(self, pipe_info):
        self.pipe_info = pipe_info
        values = pipe_info.values
        cardidx = pipe_info.cardid
        icard = cardidx % len(values.card)
        iblk = cardidx  // len(values.card)
        card = values.card[icard]
        block = values.block[iblk]
        if values.cardcap_dir is not None:
            files = glob.glob(os.path.join(values.cardcap_dir, '*.fits'))
            log.debug('Found %d cards in %s', len(files), values.cardcap_dir)
        else:
            files = values.files
            
        fstr = f'b{block:02d}_c{card:02d}'
        myfiles = sorted(filter(lambda f: fstr in f, files))
        log.info(f'Cardidx {cardidx} icard={icard} iblk={iblk} card={card} iblk={iblk} has {len(myfiles)} files ')
        assert len(myfiles) == NFPGA, f'Incorrect number of cards for cardidx={cardidx} {fstr} {myfiles} {files} '
        self.merger = CcapMerger(myfiles)
        self.myfiles = myfiles
        self.values = values
        self.card = card
        self.block = block

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
    

class CardCapNetworkSource:
    def __init__(self, pipe_info):
        block_cards  = []
        procid = 0
        cardno = 0
        values = pipe_info.values
        numprocs = pipe_info.rx_comm.Get_size()

        # assign all FPGAs to each rank
        for blk in values.block:
            for crd in values.card:
                cardno += 1
                block_cards.append((blk, crd, values.fpga))
                procid += 1
                if procid > numprocs:
                    break

        self.ctrl = cardcap.MpiCardcapController(pipe_info.rx_comm,
                                                  pipe_info.values,
                                                  block_cards)
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
        self.fid0 = self.merger.get_fid0_from_start_bat(self.start_bat)
        
        return self.fid0

    def packet_iter(self):
        assert self.fid0 is not None, 'Havent called start'
        iters = [frame_id_iter(cap.packet_iterator(), self.fid0) for cap in self.ctrl.ccap.fpga_cap]
        while True:
            try:
                fpga_data = [next(fiter) for fiter in iters]
                packets = [fd[1] for fd in fpga_data]
                fids = [fd[0] for fd in fpga_data]
                yield (packets, fids)
            except StopIteration:
                break

        self.ctrl.stop()
                

def open_source(pipe_info):
    values = pipe_info.values
    if values.cardcap_dir is not None:
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
    def __init__(self, data, iblk, info, cas=None, ics=None):
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
        Returns  a length nbl array of bool. True if antenna is flagged
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
