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
import datetime
import craco.leapseconds as leapseconds
from craco.utils import time_blocks, bl2ant
import craco.cmdline as cmdline
import craco.aktime as aktime


__author__ = "Keith Bannister <keith.bannister@csiro.au>"

# Header format see;https://confluence.csiro.au/display/CRACO/Correlator+packet+format
# BAT, FREQ, BeamID, PolID, reserved, reserve
HEADER_FORMAT = "<QIBBxx"

class CorrelatorSimulator:

    def __init__(self, values):
        fin = values.uvfits[0]
        self.hdu = fits.open(fin, mode='readonly')
        self.hdu.info()
        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_DGRAM) # UDP socket
        self.dest_address = values.destination
        self.beamid = values.beamid
        assert 0 <= self.beamid < 36, 'Invalid Beam ID'
        
        self.polid = values.polid
        for p in self.polid:
            assert 0 <= p < 4, f'Invalid Pol ID:{p}'

        self.tsamp = values.tsamp
        assert 0 < self.tsamp, f'Invalid tsamp:{selftsamp}'

        self.nloops = values.nloops
        self.nsent = 0
        
        logging.info('Sending data from %s to %s with beamID=%d and polID=%s tsamp=%dus',
                     fin, self.dest_address, self.beamid, self.polid, self.tsamp)

        
    @classmethod
    def from_args(cls, values):
        '''
        Build a correlator simulator from command line arguments
        '''
        return CorrelatorSimulator(values)

    def send(self, message):
        self.sock.sendto(message, self.dest_address.hostport)
        self.nsent += 1

    def send_block(self, iblk, blk, bat):
        '''
        Builds packets and sends them

        :iblk: Block id (starts at zero)
        :blk: Block data
        :bat: Binary atomic time
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
        for c in range(nchan):
            for polid in self.polid:
                freq = float(c)
                beamid = self.beamid
                hdr = struct.pack(HEADER_FORMAT, bat, c, beamid, polid)
                packet = hdr + data[:, c].tobytes()
                self.send(packet)
            
    def run(self):
        '''
        Run the simulator - it might return or it might not
        '''
        d = self.hdu['PRIMARY'].data
        utc_now = aktime.utc_now()
        dutc_now = leapseconds.dTAI_UTC_from_utc(utc_now.replace(tzinfo=None)).total_seconds()
        bat_now = aktime.utcDt2bat(utc_now, dutc=dutc_now)
        jd_now = aktime.bat2utc(bat_now, dutc=dutc_now)
        utcdt_back = aktime.bat2utcDt(bat_now)
        logging.info('First time is UTC=%s utc_back=%s dutc=%d bat=0x%x jd=%f ', utc_now.isoformat(),
                      utcdt_back,
                     dutc_now, bat_now, jd_now)

        bat_blk = bat_now

        for iloop in range(self.nloops):
            for iblk, blk in enumerate(time_blocks(d, nt=1)):
                self.send_block(iblk, blk, bat_blk)

            bat_blk += self.tsamp

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Simulate the ASKAP Correlator in CRACO mode', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--destination', help='Desintation UDP adddress:port e.g. localhost:1234', required=True, type=cmdline.Hostport)
    parser.add_argument('-b','--beamid', help='Beamid to put in headers, 0-35', default=0, type=int)
    parser.add_argument('-p','--polid', help='PolIDs. If input file has 1 polarisation, it duplicates it with given pol IDs', default=[0, 1], type=cmdline.strrange)
    parser.add_argument('-t','--tsamp', help='Sampling interval in microseconds', default=1728, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-n','--nloops', type=int, help='Number of times to loop through the file', default=1)
    parser.add_argument(dest='uvfits', nargs=1, type=str)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    sim = CorrelatorSimulator.from_args(values)
    sim.run()
    print(f'Sent {sim.nsent} packets')

    

if __name__ == '__main__':
    _main()
