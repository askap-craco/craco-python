#!/usr/bin/env python
"""
Print cardcap headers nicely

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco.cardcap import CardcapFile

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--type', help='Chose what type of header you want to show', choices=('timestamp','fits'), default='fits')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if values.type == 'timestamp':
        print('filename len syncbat npkt frame_id bat')
        
    first_fid = None
    first_bat = None
    for f in values.files:
        cc = CardcapFile(f)

        if values.type == 'fits':
            for hcard in cc.mainhdr.cards:
                print(f'{f} {hcard}'.strip())

        else:
            s = f"{f} {len(cc)} 0x{cc.syncbat:x} {cc.sbid}/{cc.scanid} {cc.target} {cc.mjd0} {cc.nant} {cc.nbeam} {cc.npol} {cc.dtype['data'].shape} {len(cc)} "

            try:
                f1 = next(cc.packet_iter())
                fid = int(f1['frame_id'])
                bat = int(f1['bat'])
                if first_fid is None:
                    first_fid = fid
                    first_bat  = bat
                    
                s += f"{fid} {fid-first_fid} {bat} {hex(bat)} {bat-first_bat} {fid % 2048}"
                    
            except StopIteration:
                s += 'EMPTY'
        
            print(s)
        


    

if __name__ == '__main__':
    _main()
