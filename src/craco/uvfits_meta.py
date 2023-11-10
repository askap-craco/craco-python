#!/usr/bin/env python
"""
Wraps a uvfits file but supplies flags and UVW from an ASKAP metadata file


Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import os
import sys
import logging
from craco.metadatafile import MetadataFile, to_uvw, uvw_to_array
from craft import uvfits
from craft.craco import ant2bl,bl2ant
from astropy.io import fits
from scipy import constants
import astropy.units as u

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class UvfitsMeta(uvfits.UvFits):
    def __init__(self, hdulist, max_nbl=None, mask=True, skip_blocks=0, metadata_file=None):
        self.meta_file = MetadataFile(metadata_file)
        super().__init__(hdulist, max_nbl, mask, skip_blocks)

    @property
    def baselines(self):
        tstart = self.tstart
        tuvw = tstart
        beamid = self.beamid
        uvw = self.meta_file.uvw_at_time(tuvw)[:, beamid, :] / constants.c
        bl = super().baselines
        blout = {}
        for blid in bl.keys():
            a1,a2 = bl2ant(blid)
            ia1, ia2 = (a1 - 1), (a2 - 1)
            bluvw = uvw[ia1,:] - uvw[ia2,:]
            bluvw2 = to_uvw(bluvw)
            origuvw = uvw_to_array(bl[blid])
            do_check = np.any(origuvw != 0)
            
            if do_check:
                assert_allclose(bluvw, origuvw, rtol=5e-7)
                
            blout[blid] = bluvw2
            
        return blout

def open(*args, **kwargs):
    return UvfitsMeta(fits.open(*args, **kwargs), **kwargs)

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
