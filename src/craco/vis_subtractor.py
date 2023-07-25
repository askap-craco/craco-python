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

class VisSubtractor:
    '''
    Configurable block-based visibility subtractor
    The average is the final axis

    If the subtraciton block ==teh block size, it will subtract the average of the block from itself

    If the subtraction block is < the block size, it will subtrac the sub blocks from themselves

    If the subtraction block is > the block size, it will collect an average and subtract that average from future blocks
    '''

    def __init__(self, input_shape, subtract_nt, dtype=np.complex64):

        nt = input_shape[-1]
        assert subtract_nt > 0
        assert nt > 0
        if subtract_nt < nt:
            if  nt % subtract_nt != 0:
                raise ValueError('Subtraction nsamp must divide evenly into nt')
        else:
            if subtract_nt % nt != 0:
                raise ValueError('nt must divide evenly into nt')

        self.input_shape = input_shape
        self.subtract_nt = subtract_nt
        self.nt = nt
        if self.subtract_nt > self.nt:
            sum_shape = self.input_shape.copy()
            sum_shape[-1] = 1 # this dimsions is averaged over
            self.curr_sum = np.zeros(sum_shape, dtype=dtype) # allocate buffer doesnt include final axis
            self.curr_avg = self.curr_sum.copy()
        else:
            self.curr_sum = None

        self.iblk = 0
        self.nsum = 0

    def __call__(self, input_flat):
        if self.subtract_nt == self.nt: # Subtract average of current block
            input_flat -= input_flat.mean(axis=-1, keepdims=True)
        elif self.subtract_nt < self.nt: # calculate average of sub-blocks
            sz = self.subtract_nt
            nblocks = self.nt // sz
            for blk in range(nblocks):
                start = blk*sz
                end = start + sz
                input_flat[...,start:end] -= input_flat[...,start:end].mean(axis=-1, keepdims=True)
        else: # average over several blocks
            assert self.subtract_nt > self.nt
            self.curr_sum += input_flat.sum(axis=-1, keepdims=True)
            self.nsum += input_flat.shape[-1]
            nblk = self.subtract_nt // self.nt

            # update rescaling for first block but only reset every N blocks thereafter
            if self.iblk == 0 or self.iblk % nblk == (nblk - 1):
                self.curr_avg = self.curr_sum / self.nsum

                if self.iblk > 0: 
                    self.curr_sum[:] = 0
                    self.nsum = 0
                
            input_flat -= self.curr_avg

        self.iblk += 1

        return input_flat

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
