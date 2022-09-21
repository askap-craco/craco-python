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
    parser.add_argument('-o','--output', help='Output file name')
    parser.add_argument('-t','--threshold', help='Cut candidates below this threshold',  type=float)
    parser.add_argument('-c','--maxcount', help='Maximum number of rows to load', type=int)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dtype = np.dtype([('SNR',np.float32),
                      ('lpix', np.uint16),
                      ('mpix', np.uint16),
                      ('boxc_width', np.uint8),
                      ('time', np.int),
                      ('dm', np.int),
                      ('iblk', np.int),
                      ('rawsn', np.int),
                      ('total_sample', np.int),
                      ('obstime_sec', np.float32),
                      ('mjd', np.float64),
                      ('dm_pccm3', np.float32),
                      ('ra_deg', np.float64),
                      ('dec_deg', np.float64)])
                      
                      
    for f in values.files:
        c = np.loadtxt(f, dtype=dtype, max_rows=values.maxcount)
        if values.threshold is not None:
            c = c[c['SNR'] >= values.threshold]
            
        fig, ax = pylab.subplots(2,2)
        dmhist = ax[0,0]
        snhist = ax[0,1]
        candvt = ax[1,0]
        candimg = ax[1,1]

        snhist.hist(c['SNR'], histtype='step', bins=50)
        snhist.set_xlabel('S/N')
        snhist.set_ylabel('count')

        dmhist.hist(c['dm_pccm3'], histtype='step', bins=50)
        dmhist.set_xlabel('DM (pc/cm3)')
        dmhist.set_ylabel('count')
        

        candvt.scatter(c['obstime_sec'], c['dm_pccm3']+1, c['SNR'])
        candvt.set_yscale('log')
        candvt.set_xlabel('Obstime (sec)')
        candvt.set_ylabel('1+DM (pc/cm3)')

        candimg.scatter(c['ra_deg'],c['dec_deg'], c['SNR'])
        candimg.set_xlabel('RA (deg)')
        candimg.set_ylabel('Dec (deg)')

        fig.tight_layout()
        pylab.show()
        if values.output:
            pylab.savefig(values.output)
        
    

if __name__ == '__main__':
    _main()
