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

def cmplxplot(d, title='', xlabel='', ylabel=''):
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
    ax[1,0].set_xlabel(xlabel)
    ax[1,1].set_xlabel(xlabel)

    ax[0,0].set_ylabel(ylabel)
    ax[1,0].set_ylabel(ylabel)


    return fig,ax


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-m','--metadata-file', help='Metadatda file')
    parser.add_argument('--calc11', action='store_true')
    parser.add_argument('-b','--bl', type=int, default=0, help='Baseline to plot')
    parser.add_argument('-c','--chan', type=int, default=0, help='Chan to plot')
    parser.add_argument('-t','--sample', type=int, default=0, help='sample to plot')
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
    chan = values.chan
    samp = values.sample
    bl = values.bl
    # d.shape = (nbl, nf, nt)
    cmplxplot(d[:,chan,:], title=f'Channel {chan}', xlabel='t', ylabel='bl') # channel
    cmplxplot(d[:,:,samp], title=f'Sample {samp}', xlabel='chan', ylabel='bl') #  sample
    cmplxplot(d[bl,:,:], title=f'Baseline {bl}', xlabel='chan', ylabel='t') # baseline

    cmplxplot(d.mean(axis=2), title='sample average', xlabel='chan', ylabel='bl')


    pylab.show()

    

if __name__ == '__main__':
    _main()
