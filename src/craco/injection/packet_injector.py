#!/usr/bin/env python
"""
Injects FRBs into pckets on a card level so you can have more fun.

Copyright (C) CSIRO 2025
"""
import numpy as np
import os
import sys
import logging
from numba import njit, prange
from craco.card_averager import beamchan2ibc,ibc2beamchan

from craco.cardcapfile import NCHAN, NFPGA, get_indexes, NBEAM, debughdr
from craco.mpi_obsinfo import MpiObsInfo
from craco.card_averager import Averager
from craco.injection.injection_generator import get_injector


__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)


@njit(fastmath=True, boundscheck=True, parallel=True, cache=True)
def inject(packets, valid, isamp, phasors, chan_delays, tfmap):
    '''
    input: List of NFPGA packets. 1 per FPGA
    Each packet is a (144, nt1) shaped list of packets. 144 is the 36*4 course channels in IBC order
    The packet contains a header and 'data' which is shape (nt1, nbl), which assumes npol=1 and comlpex data dtype.
     where nbl includes autos for 30 
    Or maybe we make 'data' (nt1, nbl) and a complex view with the npol scrunched. Yeah that's it.

    valid is a list of NFGPA bools. True if data are valid. False otherwise.

    isamp is the sample number of the first sample of the packets - useful for working out when to inject the pulse
    phasors is (nbeam, nbl) (including autos) complex phasors to inject the FRB at
    chan_delays is (nbeam, nchan) sample delays (with respect to zero)
    tfmap is (nfburst, ntburst) FRB time-frequency delay, with most of the delays removed (so it's small)

    '''
    nfpga = len(packets)
    nfburst, ntburst = tfmap.shape

    for ifpga in prange(len(packets)):
        if not valid[ifpga]:
            continue
        pkt = packets[ifpga]
        
        for ibc in range(pkt.shape[0]):
            ibeam, ichan = ibc2beamchan(ibc)
            ochan = ifpga + ichan*NFPGA
            nt1 = pkt.shape[1]
            for t1 in range(nt1):
                d = pkt[ibc, t1]['data']
                nt2 = d.shape[0]
                for t2 in range(nt2):
                    itnow = t1*nt2 + t2
                    tnow = isamp + itnow
                    tfidx = tnow - chan_delays[ibeam, ochan]
                    nbl = d.shape[1]
                    if 0 <= tfidx < ntburst and tfmap[ochan,tfidx] != 0:  
                        amp = tfmap[ochan, tfidx]                      
                        for ibl in range(nbl):
                            frbval = amp * phasors[ibeam, ibl] # complex value
                            d[t2,ibl,0,0] += np.int16(frbval.real)
                            d[t2,ibl,0,1] += np.int16(frbval.imag)
    



class PacketInjector:
    def __init__(self, card_averager:Averager, obs_info:MpiObsInfo):
        self._card_averager = card_averager
        self._obs_info = obs_info
        self._nt = self._obs_info.nt        
        self._isamp = 0

        nbeam = 36
        nc = NFPGA*NCHAN
        nbl = 30*(30+1)//2 # includes autos
        amplitude = 1
        self.injector = get_injector(self._obs_info)
        self.injection = next(self.injector)


    def inject(self, data, valid):
        '''
        Injects some FRB data into the packets, if needed
        '''

        log.info('Injecting into shape %s isamp=%s nt=%s phasors=%s chan_delays=%s tfmap=%s', 
                 data[0]["data"].shape, self._isamp, self._nt,
                 self.injection.phasors.shape, 
                 self.injection.chan_delays.shape, 
                 self.injection.tfmap.shape)

        inject(data, valid, self._isamp, 
               self.injection.phasors, 
               self.injection.chan_delays, 
               self.injection.tfmap)        

        self._isamp += self._nt
        if self._isamp > self.injection.end_samp:
            self.injection = next(self.injector)            


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
