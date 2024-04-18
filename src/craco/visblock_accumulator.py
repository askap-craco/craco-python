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
from craco.timer import Timer


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class VisblockAccumulator:
    def __init__(self, nbl, nf, nt):
        shape = (nbl, nf, nt)
        self.pipeline_data = np.ma.masked_array(np.zeros(shape, dtype=np.complex64), mask=np.zeros(shape, dtype=bool))
        self.t = 0
        self.nt = nt

    def write(self, vis_block):
        vis_data = vis_block.data
        assert len(vis_data.shape) >= 4, f'Invalid vis data shape {vis_data.shape} {vis_data.dtype}'
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert vis_data.dtype == np.complex64, f'I think we can only handle complex data in this function. Vis data type was {vis_data.dtype} {vis_data.shape}'
        output_nt = self.pipeline_data.shape[2]
        
        assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
        assert vis_nc*nrx == self.pipeline_data.shape[1], f'Output NC should be {self.pipeline_data.shape[1]} but got {vis_nc*nrx} {vis_data.shape}'
        assert self.pipeline_data.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'
        blflags = vis_block.baseline_flags[:,np.newaxis,np.newaxis]
        tstart = self.t
        tend = tstart + vis_nt

        # loop through each card
        for irx in range(nrx):
            fstart = irx*vis_nc
            fend = fstart + vis_nc
            chanmask = abs(vis_data[irx, ...]) == 0
            self.pipeline_data[:,fstart:fend, tstart:tend] = vis_data[irx, ...]
            self.pipeline_data.mask[:,fstart:fend, tstart:tend] = chanmask | blflags

        self.t += vis_nt
        output_nt = self.pipeline_data.shape[2]

    @property
    def msg_size(self):
        # 1 byte for mask plus a bit for extra
        s = self.pipeline_data.size * (self.pipeline_data.itemsize + 1) + 1024 
        return s

    @property
    def is_full(self):
        return self.t == self.nt
    
    def reset(self):
        self.t = 0

    def close(self):
        pass







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
