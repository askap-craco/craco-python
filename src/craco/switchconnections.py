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

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    startport = 27
    i = 0
    for blk in range(2, 7+1):
        for card in range(1, 12+1):
            for fpga in ('135','246'):
                port = startport + i // 4
                lane = i % 4
                print(f'Ethernet1/{port}/{lane+1} b{blk} c{card:02d} f{fpga}')
                i += 1
                
    
    

if __name__ == '__main__':
    _main()

    
