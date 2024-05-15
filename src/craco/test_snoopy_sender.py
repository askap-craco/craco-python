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
import pytest
from craco.candidate_writer import CandidateWriter
from pytest import fixture
from craco.snoopy_sender import *
import socket
log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


cand = np.zeros(1, dtype=CandidateWriter.out_dtype)[0]
cand['snr'] = 12.0
cand['total_sample'] = 32
cand['obstime_sec'] = 32*.110
cand['boxc_width'] = 3
cand['dm'] = 12
cand['dm_pccm3'] = 12345.1
cand['ibeam'] = 12

def test_format():
    sender = SnoopySender()
    s = sender.format_candidate(cand)
    print(s)

def test_void_to_dict():
    cout = np_void_to_dict(cand)

def test_send():
    host = '127.0.0.1'
    port = 12345
    sender = SnoopySender(host, port)
    
    #sock.setsockopt(socket.SO_REUSEADDR)
    cand_str = sender.format_candidate(cand)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sender.send(cand)
        d = sock.recv(1024).decode('utf-8')
        print(d)
        assert d == cand_str, 'Expected {d} == {cand_str}'
    finally:
        sock.close()


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
