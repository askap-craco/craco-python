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

ip_address= 'ff02:0:0:c985::'
udp_port = 4792
sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)

def sender(values):
    msgno = 0
    hostport = (values.ip_address, values.udp_port)
    while True:
        msg = f"Message number {msgno} to {hostport}"
        print(f'Sending {msg} to {hostport}')
        sock.sendto(msg.encode('utf-8'), hostport )
        time.sleep(1)


def receiver(values):
    #hostport = (values.ip_address, values.udp_port)
    hostport = ('', values.udp_port)
    sock.bind(hostport)

    while True:
        data, addr = sock.recvfrom(1024)
        dd = data.decode('utf-8')
        print(f'Got {addr} {dd}')
    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument("--sender", action='store_true', default=False)
    parser.add_argument("-a", "--ip-address", default=ip_address)
    parser.add_argument("-p", "--udp-port", default=udp_port, type=int)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if values.sender:
        sender(values)
    else:
        receiver(values)
    

if __name__ == '__main__':
    _main()
