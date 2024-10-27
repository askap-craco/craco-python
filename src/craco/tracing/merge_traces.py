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
import glob


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-N', '--maxn', help='Max number of lines per file', type=int, default=-1)
    parser.add_argument(dest='files', nargs='*')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    cwd = os.getcwd()
    cwdbits = cwd.split('/')
    cwdbits[1] = 'CRACO'
    cwdbits[2] = 'DATA*'
    globpath = '/'.join(cwdbits) + '/rank*trace.json'
    dirs = os.getcwd().split('/') # '/data/SKADI_00_0/craco/SB067194/scans/00/20241027232156'
    sbid = dirs[4]
    scanid = dirs[-1]
    foutname = f'{sbid}-{scanid}_traces.json'
    fout = open(foutname, 'wt')
    # assumes array mode
    fout.write('[\n')
    files = glob.glob(globpath)
    log.info('Got %d files in %s', len(files), globpath)
    oline = 0
    for f in files:
        with open(f, 'rt') as fin:
            for iline, line in enumerate(fin):
                if values.maxn > 0 and iline -1 == values.maxn:
                    break

                if line.startswith('[') or line.startswith(']'):
                    continue
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                if oline > 0:
                    line = ',\n'+line

                fout.write(line)
                oline += 1
        

    fout.write('\n]\n')
    fout.close()
    log.info('Wrote %s %d lines %d bytes', os.path.abspath(foutname), oline, os.path.getsize(foutname))



    

if __name__ == '__main__':
    _main()
