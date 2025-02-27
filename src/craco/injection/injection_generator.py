#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2025
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.cardcap import CardcapFile, get_single_packet_dtype, NCHAN,NFPGA

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)

class Injection:
    def __init__(self, phasors, chan_delays, tfmap):
        self.phasors = phasors
        self.chan_delays = chan_delays
        self.tfmap = tfmap
        self.end_samp = self.chan_delays.max() + self.tfmap.shape[1]


def dm0_pc_regular(nbeam:int, nbl:int, nc:int, delta_t:int=256, delta_beam:int=1, amplitude:float=1, start_t:int=0):
    '''
    Must return 465 baseliens (include autos)
    nc=24 = NFPGA*NCHAN (fine channels per card)
    
    Phase center
    DM0 only
    Every delta_t samples
    Offset every beam by delta_beam in samples
    '''
    # everything on the phase center
    phasors = np.ones((nbeam, nbl), dtype=np.complex64)

    # Beam-dependent channel delays
    chan_delays = np.zeros((nbeam, nc), dtype=np.int64) # global channel days. Add DM sweep plus global start time to this.
    
    # DM 0 pulse at t=0
    tfmap = np.zeros((nc, 1), dtype=np.float32) 
    tfmap[::,0] = amplitude
    
    tstart = 0
    while True:
        for ibeam in range(nbeam):
            chan_delays[ibeam,:] = tstart + ibeam*delta_beam + start_t

        yield Injection(phasors, chan_delays, tfmap)
        tstart += delta_t


def get_injector(info):
    return dm0_pc_regular(36, 465, NFPGA*NCHAN, delta_t=256, delta_beam=1, amplitude=10*3, start_t=247)

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
