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

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class CorrelatorSimulator:
    pass

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Simulate the ASKAP Correlator in CRACO mode', formatter_class=ArgumentDefaultsHelpFormatter)
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
