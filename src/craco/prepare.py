#!/usr/bin/env python
"""
Prepare data from FPGAs suitable for transposing

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
    '''Generator replicates the crazy ordering of beams and channels in the ASKAP correlator
    Yields (channel, beam) pairs
    Zero indexed
    @param beam_mask contains 1 if you want that beam. Zero otherwise.
    '''
    slot = 0
    for c in range(4):
        for b in range(32):
            if (beam_mask >> b) & 0x1 == 1:
                yield (c, b, slot)

            slot += 1

    for c in range(4):
        for b in range(32, 36):
            if (beam_mask >> b) & 0x1 == 1:
                yield (c, b, slot)

            slot += 1
            


@jit(nopython=True)
def do_prepare_numba(input_data, channel_map, calibration_data, sky_model, output_data, block_average, scale=1.0, beam_mask=0xfffffffff, check=False):
    '''
    Prepares input data
    @param input data = list numpy arrays. length of the list is the number of FPGAS of interest (nfpga)
    Each input data has shape  = (ncoarse_chan*nbeam, nt_per_bf_frame, nfine_chan, nbl, 2) np.int16
    where ncoarse_chan = 4, nbeam = 5, nt_per_bf_frame is variable (2048 // 16, usually), nfine_chan = 6 and  nbl is configurable
    every array should have the same shape
    It's a funny shape because ..... digital design reasons.

    @param channel_map array length of nfpga*ncoarse_channels*nfine_channls channels that permutes the input to the output mapping
    @param calibration_data - multiply input but this 
    @param sky_model - subtract this from calibrated data
    @output_data suitable for transposing shape = [nbeams, nfpga*ncoarse_channels*nfine_channels
    @block_average averaged data

    '''

    # how to you make a float32 scalar in numba???? 
    fscale = np.array([scale], dtype=np.float32)
    nfpga = len(input_data)
    total_nchan = nfpga*ncoarse_channels*nfine_channels
    inshape = input_data[0].shape
    nctimesb, nt, nfine_chan, nbl, expect2 = inshape


    if check:
        assert expect2 == 2
        assert nfine_chan == nfine_channels
        assert nctimesnb == ncoarse_channels*nbeams
        assert inshape == np.int16

        assert calibration_data.shape == (total_nchan, nbl)
        assert calibration_data.dtype == np.complex64

        assert sky_model.shape == (total_nchan, nbl)
        assert sky_model.dtype == np.complex64

        assert block_average.shape == (total_nchan, nbl)
        assert block_average.dtype == np.complex64

    for ifpga, input_data in enumerate(input_data):
        # create complex 64 view without making a copy (I think?). Remove last dimension as it's aborbed into complex
        incomplex = input_data.view(dtype=np.complex64)[...,0]
        for islot, (coarse_channel, beam, cbslot) in enumerate(coarse_beam_gen(beam_mask)):
            for t in prange(nt):
                for fine_channel in prange(nfine_chan):
                    total_channel = ifpga*ncoarse_channels*nfine_channels + coarse_channel*nfine_channels + fine_channel
                    out_channel = channel_map[total_channel]
                    ind = incomplex[cbslot, t, :]
                    block_average[total_channel, :] += ind  # calculate average over time as complex64
                    
                    # calculate output in complex64
                    dout = ind*calibration_data[total_channel,:] - sky_model[total_channel, :]
                    output[beam, total_channel, t,:] = dout*fscale # cast to int16

    return output_data, block_average
        

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


    nfpga = 6*12
    data = []
    nt = 128
    nbl = 435
    dshape = (nbeams*ncoarse_channels,  nt, nfine_chanels, nbl, 2)
    for f in range(nfpga):
        data.append(np.arange(np.prod(dshape), dtype=np.int16).reshape(dshape))

    ntotal_nchan = nfpga*ncoarse_channels*nfine_channels
    calibration_data = np.zeros((total_nchan, nbl), dtype=np.complex64)
    skymodel_data = calibration_data.copy()
    block_average = skymodel_data.copy()
    calibration_data[:] = 1
    channel_map = np.arange(total_nchan, dtype=np.int32)
    output_data = np.zeros((nbeam, total_nchan, nt, nbl, 2), dtype=np.int16)
    do_prepare_numba(input_data, channel_map, calibration_data, sky_model_data, output_data, block_average, scale=1.0, beam_mask=0xfffffffff, check=False)

    

    

if __name__ == '__main__':
    _main()
