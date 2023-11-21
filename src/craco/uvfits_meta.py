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
from craft.vis_metadata import VisMetadata
from astropy.io import fits
from scipy import constants
import astropy.units as u
from astropy.time import Time

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class UvfitsMeta(uvfits.UvFits):
    def __init__(self, hdulist, max_nbl=None, mask=True, skip_blocks=0, metadata_file=None):
        self.meta_file = MetadataFile(metadata_file)
        super().__init__(hdulist, max_nbl, mask, skip_blocks)

    def vis_metadata(self, isamp:int):
        '''
        Return a vis info adapter for the given sample number
        '''

        tstart = self.tstart + self.tsamp*isamp

        m = VisMetadata(
            self.baselines_at_time(tstart),
            self.freq_config,
            self.target_name,
            self.target_skycoord,
            self.beamid,
            tstart,
            self.tsamp,
            tstart)
        m.isamp = isamp

        return m
    
    @property
    def baselines(self):
        bl = self.baselines_at_time(self.tstart)
        return bl

    def baselines_at_time(self, tuvw:Time):
        '''
        Returns baselines interpolated using the metadata to a particular
        time
        '''
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
            do_check = np.any(origuvw != 0) and False
            
            if do_check:
                assert_allclose(bluvw, origuvw, rtol=5e-7)
                
            blout[blid] = bluvw2
            
        return blout

def open(*args, **kwargs):
    logging.info('Opening file %s', args[0])
    mfile = kwargs.get('metadata_file', None)
    if mfile is None:
        del kwargs['metadata_file']
        x = uvfits.open(*args, **kwargs)
    else:
        x = UvfitsMeta(fits.open(*args, **kwargs), **kwargs)

    return x


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
