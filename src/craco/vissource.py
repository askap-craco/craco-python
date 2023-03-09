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
from craco import cardcap
from craco.cardcapmerger import CcapMerger
import astropy.io.fits.header as header

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class VisSource:
    pass


class CardCapFileSource:
    '''
    Gets data from a single card - i.e. 6 FPGAs
    '''
    def __init__(self, cardidx, values):
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

    def packet_iter(self, start_fid=None):
        return self.merger.packet_iter(start_fid=start_fid, beam=self.values.beam)


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
        this_block_card = block_cards[pipe_info.rx_comm]
        self.block, self.card = this_block_card[0:2]


    @property
    def fpga_headers(self):
        # convert to proper fits header. Insane.
        
        fitshdr = header.Header()
        for k,v in self.ctrl.ccap.hdr.items():
            fitshdr[k] = v

        return [str(fitshdr)]

    def packet_iter(self):
        self.ctrl.configure_and_start()
        iters = [cap.packet_iterator() for cap in self.ctrl.ccap.fpga_cap]
        while True:
            try:
                fpga_data = [next(fiter) for fiter in iters]
                packets = [fd[1] for fd in fpga_data]
                fids = [fd[0] for fd in fpga_data]
                yield (packets, fids)
            except StopIteration:
                break
                

def open_source(pipe_info):
    values = pipe_info.values
    cardidx = pipe_info.cardid
    if values.cardcap_dir is not None:
        src = CardCapFileSource(cardidx, values)
    else:
        src = CardCapNetworkSource(pipe_info)

    return src
    

class VisBlock:
    '''
    Represents a block of data from a single source for a single beam.
    All baselines
    Block of NT integrations
    '''
    def __init__(self):
        pass

    @property
    def data(self):
        return self._d

    @property
    def uvw(self):
        # TODO: calculate if appropriate
        return self._uvw

    @property
    def mjd(self):
        return self._mjd

    @property
    def bat(self):
        return self._bat

    @property
    def beam(self):
        return self._beam

    @property
    def nt(self):
        return self._nt
    

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
