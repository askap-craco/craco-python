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
from craco import uvfits_meta

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def cmplxplot(d, title=''):
    fig, ax = pylab.subplots(2,2)
    fig.suptitle(title)
    fig.tight_layout()
    ax[0,0].imshow(d.real, aspect='auto')
    ax[0,1].imshow(d.imag, aspect='auto')
    ax[1,0].imshow(abs(d), aspect='auto')
    ax[1,1].imshow(np.angle(d), aspect='auto')

    ax[0,0].text(0,0,'real',ha='left', va='bottom')
    ax[0,1].text(0,0,'imag',ha='left', va='bottom')
    ax[1,0].text(0,0,'abs',ha='left', va='bottom')
    ax[1,1].text(0,0,'ang',ha='left', va='bottom')


    return fig,ax


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-m','--metadata-file', help='Metadatda file')
    parser.add_argument('--calc11', action='store_true')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    f = uvfits_meta.open(values.files[0])
    
    inf = uvfits_meta.open(values.files[0], 
                           metadata_file=values.metadata_file, 
                           calc11=values.calc11)
    
    nt = 256
    d, uvw = next(inf.fast_time_blocks(nt, fetch_uvws=True, istart=0))
    d = d.squeeze()
    print(type(d), d.shape, type(uvw))
    # d.shape = (nbl, nf, nt)
    cmplxplot(d[:,0,:], title='Channel 0')
    cmplxplot(d[:,:,0], title='Sample 0')
    cmplxplot(d[0,:,:], title='Baseline 0')

    pylab.show()

    

if __name__ == '__main__':
    _main()
