#!/usr/bin/env python
"""
Prepare data from FPGAs suitable for transposing

 A note on the FPGA data

 fpga_data = list numpy arrays. length of the list is the number of FPGAS of interest (nfpga)
    Each input data has shape  = (ncoarse_chan*nbeam, nt_per_bf_frame,  nbl, 2) np.int16
    where ncoarse_chan = 4, nbeam = 36, nt_per_bf_frame is variable (2048 // 16, usually), and  nbl is configurable
    every array should have the same shape
    It's a funny shape because ..... digital design reasons we wont' go into


Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from numba import jit,prange

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

# these numbers are fixed in the FPGA packet transmission code and are unlikely to change.
ncoarse_channels=4
nfine_channels=6
nbeams=36

@jit(nopython=True)
def coarse_beam_gen(beam_mask=0xfffffffff):
    '''
    Generator replicates the crazy ordering of beams and channels in the ASKAP correlator
    yields tuples (coarse_channel, outbeam, inbeam, cbslot)
    Zero indexed
    where
    coarse_channel is the coarse_channel index (0-3)
    outbeam is the output beam index (0 - (the number of ones in the mask))
    inbeam - the input beam index (0-36)
    cbslot - the channel*slot index
    @param beam_mask contains 1 if you want that beam.

    >>> list(coarse_beam_gen(0x1))
    [(0, 0, 0, 0), (1, 0, 0, 32), (2, 0, 0, 64), (3, 0, 0, 96)]

    >>> list(coarse_beam_gen(0x13))
    [(0, 0, 0, 0),
    (0, 1, 1, 1),
    (0, 2, 4, 4),
    (1, 0, 0, 32),
    (1, 1, 1, 33),
    (1, 2, 4, 36),
    (2, 0, 0, 64),
    (2, 1, 1, 65),
    (2, 2, 4, 68),
    (3, 0, 0, 96),
    (3, 1, 1, 97),
    (3, 2, 4, 100)]

    >>> list(coarse_beam_gen(0x180000000))
    [(0, 0, 31, 31),
    (1, 0, 31, 63),
    (2, 0, 31, 95),
    (3, 0, 31, 127),
    (0, 1, 32, 128),
    (1, 1, 32, 132),
    (2, 1, 32, 136),
    (3, 1, 32, 140)]
    '''
    slot = 0

    for c in range(4):
        outbeam = 0
        for b in range(32):
            if (beam_mask >> b) & 0x1 == 1:
                yield (c, outbeam, b, slot)
                outbeam += 1

            slot += 1

    end_outbeam = outbeam

    for c in range(4):
        outbeam = end_outbeam
        for b in range(32, 36):
            if (beam_mask >> b) & 0x1 == 1:
                yield (c, outbeam, b, slot)
                outbeam += 1

            slot += 1


def freqavg(fpga_data: list, channel_map: np.ndarray, output_data: np.ndarray, beam_mask:int =0xfffffffff):
    '''
    Frequency averaging and re-ordering of the data directly from the FPGAs into output_data that can
    be corner-turned over MPI. Each card comprieses 6 FPGAS. 
    
    @param fpga_data list of nump arrays from the fpgas. length=multiple of 6
    @param channel_map numpy array shape[nfpga, ncoarse_channels] that maps the channel from the given FPGA into the 
    coarse channel index the output
    @output_data suitable for transposing shape = [nbeams_out, nfpga*ncoarse_channels, nt, nbl, 2]
    @block_average averaged data

    '''

    nfpga = len(fpga_data)
    nchan_input = nfpga*ncoarse_channels*nfine_channels
    nchan_output = nfpga*ncoarse_channels
    
    inshape = fpga_data[0].shape
    nctimesnb, nt, nbl, expect2 = inshape
    nbeams_out = output_data.shape[0]


    assert expect2 == 2
    assert nctimesnb == ncoarse_channels*nbeams
    assert channel_map.shape== (nfpga, ncoarse_channels)
    assert output_data.shape[1:] == (nchan_output, nt, nbl, 2)


    for ifpga, indata in enumerate(fpga_data):
        for coarse_channel, outbeam, inbeam, cbslot in coarse_beam_gen(beam_mask):
            outchan = channel_map[ifpga, coarse_channel]
            output_data[outbeam, outchan, :, :,:] += indata[cbslot, :, :, :]

    return output_dataw

def rescale(input_data, output_data, scale=1):
    '''
    Rescale the input data into the output data
    Useful for compressing the output to reduce network bandwidth
    '''
    output_data[:] = (input_data // scale)
