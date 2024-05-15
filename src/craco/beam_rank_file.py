#!/usr/bin/env python
"""
Make a beam rank file from an mpipipeline rank file
Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
import re

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

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

    for f in values.files:
        with open(f, 'r') as fin:
            rank = 0
            for line in fin:
                if 'xrtdevid' in line:
                    bits = line.split('=',1)
                    print(f'rank {rank}={bits[1]}'.strip())
                    rank += 1

    
    

if __name__ == '__main__':
    _main()
