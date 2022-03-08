#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2018
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

from numba import jit
@jit(nopython=True)
def do_prepare_numba(input_data, calibration_data, sky_model, output_data, block_average):
    # make output data using numpy broadcasting rules - cheating slightly, but you get the idea
    nbl, nchan, npol, nt = input_data.shape
    for ibl in range(nbl):
        for ichan in range(nchan):
            for ipol in range(npol):
                for it in range(nt):
                    d = input_data[ibl, ichan, ipol, it] # read input sample
                    
                    # Output is the sum over polarisations after applying the calibration and subtracting the sky model
                    # Note these are complex multiplications and sums
                    output_data[ibl, ichan, it] += d * calibration_data[ibl, ichan, ipol] - sky_model[ibl, ichan, ipol ]
                    
                    # block average is the average over all times
                    block_average[ibl, ichan, ipol] += d
    
    return (output_data, block_average)


class Prepare(object):
    '''
    Parepares the coherent processing pipeline - applies calirbation, subtract skymodel,
    do  polarisation sum and compute average.

    In the pipeline the CPU will supply the sky mode, and calibration coefficients.
    The output will be passed on to the next stage in the pipeline
    The average is a (nbl, nchan, npol) complex numpy array will be read back by the CPU for further processing.
    The average is reset for every block.
    '''
    def __init__(self, nt, calibration_coef, sky_model):
        '''Create a prepare object
        
        :nt: Block size to expect (only used for checking)
        :calibration_coef: (nbl, nchan, npol) shape np array (complex typed) for calibration coefficients
        :sky_model: (nbl, nchan, npol) shape np array (complex typed) for sky model
        '''
        nbl, nchan, npol = calibration_coef.shape
        assert sky_model.shape == (nbl, nchan, npol), 'Sky model has incorrect shape'
        assert np.iscomplexobj(calibration_coef), 'Calibration must be complex'
        assert np.iscomplexobj(sky_model), 'Sky model must be complex'
        assert nt > 0, 'Invalid nt'

        self.nt = nt
        self.calibration_coef = calibration_coef
        self.sky_model = sky_model
        self.block_average = np.zeros((nbl, nchan, npol), dtype=calibration_coef.dtype)

    def do_prepare(self, input_data):
        ''' Actually comptue the prepared data
        updates self.block_average to the average over time

        :input_data: (nbl, nchan, npol, nt) shaped complex array of visibilities
        :returns: (nbl, nchan, nt) shaped complex array of prepared visibilities

        '''
        nt = self.nt
        nbl, nchan, npol = self.calibration_coef.shape
        assert input_data.shape == (nbl, nchan, npol, nt), 'Invalid shape for input block'

        output_data = np.empty((nbl, nchan, nt), dtype=self.calibration_coef.dtype)

        # reset average to zero
        self.block_average[:] = 0
        do_prepare_numba(input_data, self.calibration_coef, self.sky_model, output_data, self.block_average)

        # Dont return block avaerage as it doesnt go down the pipeline
        return output_data

    def __call__(self, input_data):
        return self.do_prepare(input_data)
        

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
