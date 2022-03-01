#!/usr/bin/env python
"""
pybind11 version of the Search pipeline

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging

import craco_pybind11
import craco_pybind11.ddgrid_reader
import craco_pybind11.grid

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Pipeline:
    def __init__(self, plan):
        '''
        @param plan a craco_plan
        '''
        self.plan = plan

    def ddgrid(self, mainbuf):
        '''
        Runs DDGRID on the given mainbuf
        
        @param mainbuf: np array of shape (nuvwide, ndout, ntblk, nt, nuvwide, 2), dtype=np.int16.
        2 axis is (real, imag)
        @param output: np array of shape (ndm, nt, nuv, 2), dtype=np.int16. 2 axis is (real, imag)

        '''
        nchunk_time = plan.nchunk_time
        nparallel_uvin, nparallel_uvout, h_nparallel_uvout, lut = get_grid_lut_from_plan(plan)
        nuvrest = nparallel_uvin*2//8
        # TODO: Get from pybind11
        # TODO: check input shape
        NUVWIDE = 8
        OUTPUT_NT = 2
        OUTPUT_NUV = 2
        ncu = 4
        ndm = plan.nd
        tblk = 0
        do_load_lut = 1
        outputs = np.zeros((ncu, ndm, nchunk_time, nuvrest, NUVWIDE//OUTPUT_NUV, OUTPUT_NT, OUTPUT_NUV, 2), dtype=np.int16)

        craco_pybind11.ddgrid_reader.krnl_ddgrid_reader_4cu(mainbuf, 
                                     ndm, 
                                     tblk,
                                     nchunk_time, 
                                     nuvrest, 
                                     plan.ddreader_lut, 
                                     do_load_lut, 
                                     outputs[0], 
                                     outputs[1], 
                                     outputs[2], 
                                     outputs[3])
        reordered = np.transpose(outputs, [1, 2,0,5, 3,4,6, 7])
        #assert reordered.shape == (ndm, nchunk_time, ncu, OUTPUT_NT, nuvrest, NUVWIDE, 2)
        reordered = reordered.reshape(ndm, ncu*nchunk_time*OUTPUT_NT, nuvrest*NUVWIDE, 2)

        return reordered

    def grid(self, gridbuf, icu, d_grid=None):
        '''Run GRID
        @param on the given CU
        @param d_grid if None, it will allocate the output. Otherwise , output buffer of 
        shape (ncu,ndm, nchunk_time, npix, npix, 2)
        @returns output grid
        '''
        
        
        nchunk_time = plan.nchunk_time
        nparallel_uvin, nparallel_uvout, h_nparallel_uvout, lut = get_grid_lut_from_plan(plan)
        nuvrest = nparallel_uvin*2//8
        NUVWIDE = 8
        OUTPUT_NT = 2
        OUTPUT_NUV = 2
        ncu = 4
        ndm = plan.nd
        tblk = 0
        do_load_lut = 1
        gshape = (ncu, ndm, nchunk_time, plan.npix, plan.npix, 2)
        if d_grid is None:
            d_grid = np.zeros(gshape, dtype=np.int16)

        assert d_grid.shape == gshape
        assert d_grid.dtype == np.int16

        craco_pybind11.grid.krnl_grid_4cu(ndm,
                     nchunk_time,
                     nparallel_uvin,
                     nparallel_uvout,
                     h_nparallel_uvout,
                     load_luts,
                     lut,
                     outputs[icu],
                     d_grid[icu])

        return d_grid

        
    

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
