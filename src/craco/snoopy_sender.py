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
import socket

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

DEFAULT_HOST = 'tethys'
DEFAULT_PORT = 4900


def np_void_to_dict(din):
    d = {}
    for f in din.dtype.names:
        d[f] = din[f]
    return d


class SnoopySender:
    '''
    Sends a UDP packet conforming to the old FREDDA/SNoopy candidate string so that snoopy can quit and download voltages

    Snoopy v1 is run like this:
    UDP_PORT=4900
    /home/craftop/craft/python/snoo.py 224.1.1.1:$UDP_PORT -o $snoopy_log_file --max-boxcar $max_boxcar --min-dm $snoopy_dm  --min-sn $snoopy_threshold  &/home/craftop/craft/python/snoo.py 224.1.1.1:$UDP_PORT -o $snoopy_log_file --max-boxcar $max_boxcar --min-dm $snoopy_dm  --min-sn $snoopy_threshold  &
    Snoopy v2 is run like this:
/home/craftop/craft/python/snoo_v2.py  224.1.1.1:$UDP_PORT -o $snoopyv2_log_file --max-boxcar=$max_boxcar --min-dm $snoopy_dm --min-sn $snoopy_threshold --nmin 3 --eps 7 &
snoopv2_pid=$!

# Snoopy code is here:
# https://github.com/askap-craco/craft/blob/dadain/cuda-fdmt/cudafdmt/src/snoo.py

# Snoov2 code is here:
https://github.com/askap-craco/craft/blob/master/src/craft/snoo_v2.py

# Cand format is here:
https://github.com/askap-craco/craft/blob/dadain/cuda-fdmt/cudafdmt/src/CandidateSink.cu
#define CAND_FORMAT "%0.2f %lu %0.4f %d %d %0.2f %d %0.9f\n"

                sn = npdata[:, 0]
                sampnum = npdata[:, 1]
                tsec = npdata[:, 2]
                width = npdata[:, 3]
                idt = npdata[:, 4]
                dm = npdata[:, 5]
                beamno = npdata[:, 6]
                mjd = npdata[:,7]
    '''

    CAND_FORMAT = '{snr:0.2f} {total_sample:d} {obstime_sec:0.3f} {boxc_width:d} {dm:d} {dm_pccm3:0.1f} {ibeam:d} {mjd:0.9f}\n'

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.hostport = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.IPPROTO_IP,socket.IP_MULTICAST_TTL, 30)
        #self.socket.setsockopt(socket.SOL_SOCKET. IN.SO_BINDTODEVICE, "eno8303")
        # maybe bind to this address so it goes out the right one? But it listens too?
        srcaddr = '202.9.13.200'
        #self.socket.bind((srcaddr,port))

    def format_candidate(self, cand):
        '''
        Send the given candidate in CandidateWriter.out_dtype format or anything with keys as follows
        snr
        total_sample
        obstime_sec
        boxc_width
        dm
        dm_pccm3
        mdj
        '''
        if isinstance(cand, dict):           
            dcand = cand            
        else:
            dcand = np_void_to_dict(cand)
            

        s = self.CAND_FORMAT.format(**dcand)
        return s
    
    def send(self, cand):
        cand_str = self.format_candidate(cand)
        self.socket.sendto(cand_str.encode('utf-8'), self.hostport)
        return cand_str


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--snr', type=float, default=0)
    parser.add_argument('--total_sample', type=int, default=0)
    parser.add_argument('--obstime_sec', type=float, default=0)
    parser.add_argument('--boxc_width', type=int, default=0)
    parser.add_argument('--dm_pccm3', type=int, default=0)
    parser.add_argument('--dm', type=int,default=0)
    parser.add_argument('--ibeam', type=int, default=0)
    parser.add_argument('--mjd', type=float, default='123456.789456123')

    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', default=DEFAULT_PORT, type=int)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    sender = SnoopySender(values.host, values.port)
    cand = vars(values) # makes dictionary
    s = sender.send(cand)
    log.info('Sent %s to %s', s, sender.hostport)


    

    
    

if __name__ == '__main__':
    _main()
