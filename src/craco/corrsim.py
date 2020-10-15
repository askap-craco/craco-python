#!/usr/bin/env python
"""
Correlator simulator 

Copyright (C) CSIRO 2020
"""
import numpy as np
from astropy.io import fits
import socket
import os
import sys
import logging
import socket
import struct

from craco.utils import time_blocks, bl2ant
import craco.cmdline as cmdline

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

# Header format see;https://confluence.csiro.au/display/CRACO/Correlator+packet+format
# BAT, FREQ, BeamID, PolID, reserved, reserve
HEADER_FORMAT = "<QfBBxx"

class CorrelatorSimulator:

    def __init__(self, values):
        fin = values.files[0]
        self.hdu = fits.open(fin)
        self.hdu.info()
        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_DGRAM) # UDP socket
        self.dest_address = values.destination
        self.beamid = values.beamid
        assert 0 <= self.beamid < 36, 'Invalid Beam ID'
        
        self.polid = values.polid
        assert 0 <= self.polid < 4, 'Invalid Pol ID'
        
        logging.info('Sending data from %s to %s with beamID=%d and polID=%d', fin, self.dest_address, self.beamid, self.polid)

        
    @classmethod
    def from_args(cls, values):
        '''
        Build a correlator simulator from command line arguments
        '''
        return CorrelatorSimulator(values)

    def send(self, message):
        self.sock.sendto(message, self.dest_address.hostport)

    def send_block(self, iblk, blk):
        '''
        Builds packets and sends them
        '''

        blkeys = sorted(blk.keys())
        nbl = len(blkeys)
        nchan = blk[blkeys[0]].shape[0]
        data = np.zeros((nbl, nchan), dtype=np.complex64)

        # Need to tranpose data as packets are sent on a per channel basis
        for ibl, blid in enumerate(blkeys):
            visdata = blk[blid]
            ant = bl2ant(blid)
            logging.debug('got baseline data for blid %s ant=%s shape=%s', blid, ant, visdata.shape)
            data[ibl, :] = visdata[:, 0]

        # Need to think about the baseline ordering here, but let's assume it's right for now.
        bat = 0
        for c in range(nchan):
            freq = float(c)
            beamid = self.beamid
            polid = self.polid # TODO: Support multiple polarisations in file. 
            hdr = struct.pack(HEADER_FORMAT, bat, freq, beamid, polid)
            packet = hdr + data[:, c].tobytes()
            self.send(packet)
            
    def run(self):
        '''
        Run the simulator - it might return or it might not
        '''
        d = self.hdu['PRIMARY'].data
        while True:
            for iblk, blk in enumerate(time_blocks(d, nt=1)):
                self.send_block(iblk, blk)

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Simulate the ASKAP Correlator in CRACO mode', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--destination', help='Desintation UDP adddress:port e.g. localhost:1234', required=True, type=cmdline.Hostport)
    parser.add_argument('-b','--beamid', help='Beamid to put in headers, 0-35', default=0, type=int)
    parser.add_argument('-p','--polid', help='PolID to put in headers. 0-3', default=0, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    sim = CorrelatorSimulator.from_args(values)
    sim.run()

    

if __name__ == '__main__':
    _main()
