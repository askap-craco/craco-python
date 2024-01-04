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
from craft.craco import ant2bl,bl2ant,time_block_with_uvw_range
from craft.vis_metadata import VisMetadata
from astropy.io import fits
from scipy import constants
import astropy.units as u
from astropy.time import Time

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def calc_uvw_for_blid(uvw, blid):
    a1,a2 = bl2ant(blid)
    ia1, ia2 = (a1 - 1), (a2 - 1)
    bluvw = uvw[ia1,:] - uvw[ia2,:]
    return bluvw

class VisViewMeta:
    def __init__(self, uvfits_meta):
        self.uvfits_meta = uvfits_meta
        self.offset_view = uvfits.VisView(uvfits_meta)

    @property
    def size(self):
        return self.offset_view.size

    def __getitem__(self, sidx):
        data = self.offset_view[sidx]
        all_dates = data['DATE']
        unique_dates = sorted(np.unique(all_dates))
        uvws = {}
        flags_vs_date = {}
        for d in unique_dates:
            dtime = Time(d, scale=self.uvfits_meta.tscale, format='jd')
            uvws[d] = self.uvfits_meta.uvw_array_at_time(dtime)
            flags_vs_date[d] = self.uvfits_meta.meta_file.flags_at_time(dtime)

        blids = data['BASELINE']
        dates = data['DATE']
        try:
            nbl = len(blids)
        except:
            nbl= 1
            dates = [dates]
            blids = [blids]
                
        uvw_out = np.zeros((3, nbl))
        for i, (d,blid) in enumerate(zip(dates, blids)):
            uvwarr = uvws[d]
            uvw_out[:, i] = calc_uvw_for_blid(uvwarr, blid)
            flags = flags_vs_date[d]
            if self.uvfits_meta.mask:
                a1,a2 = bl2ant(blid)
                ia1, ia2 = (a1 - 1), (a2 - 1)
                f = flags[ia1] or flags[ia2] 
                weight = 0 if f else 1
                data['DATA'][i][...,2] = weight


        data['UU'] = uvw_out[0,:]
        data['VV'] = uvw_out[1,:]
        data['WW'] = uvw_out[2,:]

        return data

class UvfitsMeta(uvfits.UvFits):
    def __init__(self, hdulist, max_nbl=None, mask=True, skip_blocks=0, metadata_file=None, start_mjd=None,end_mjd=None):
        self.meta_file = MetadataFile(metadata_file)
        super().__init__(hdulist, max_nbl, mask, skip_blocks, start_mjd, end_mjd)

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
    def vis(self):
        return VisViewMeta(self)
    
    @property
    def baselines(self):
        bl = self.baselines_at_time(self.tstart)
        return bl

    def uvw_array_at_time(self, tuvw:Time):
        '''
        Returns interpolated UVW array.
        For all baselines
        '''
        beamid = self.beamid
        uvw = self.meta_file.uvw_at_time(tuvw)[:, beamid, :] / constants.c
        return uvw

    def baselines_at_time(self, tuvw:Time):
        '''
        Returns baselines interpolated using the metadata to a particular
        time. Restricts contents of dictionary to wht's returned in superlass .baselines
        '''
        bl = super().baselines # get the baselines that are valid
        uvw = self.uvw_array_at_time(tuvw)

        blout = {}
        for blid in bl.keys():
            bluvw = calc_uvw_for_blid(uvw, blid)
            bluvw2 = to_uvw(bluvw)
            origuvw = uvw_to_array(bl[blid])
            do_check = np.any(origuvw != 0) and False
            
            if do_check:
                assert_allclose(bluvw, origuvw, rtol=5e-7)
                
            blout[blid] = bluvw2
            
        return blout
    
    def time_block_with_uvw_range(self, trange):
        """
        return a block of data and uvw within a given index range
        :trange: tuple (istart,istop) samples
        """
        d, uvw, (sstart, send) = time_block_with_uvw_range(
            vis=self.vis, trange=trange, flagant=self.flagant,
            flag_autos=self.ignore_autos, mask=self.mask
        )

        nsamp = send - sstart + 1
        assert nsamp >= 1
        for isamp in range(nsamp):
            t = self.sample_to_time(isamp + sstart)
            bl = self.baselines_at_time(t)
            for k in sorted(bl.keys()):
                uvw[k][:,isamp] = uvw_to_array(bl[k])

        return d, uvw, (sstart, send)

    @property
    def target_name(self):
        name = self.meta_file.source_name_at_time(self.tstart)
        return name

    @property
    def target_skycoord(self):
        src = self.meta_file.source_at_time(self.beamid, self.tstart)
        coord = src['skycoord']
        return coord

    def _create_masked_data2(self, dout_data, start_sampno):
        dout_complex_data = dout_data[..., 0, :] + 1j*dout_data[..., 1, :]
        mask = np.zeros(dout_complex_data.shape, dtype=bool)
        nt = mask.shape[-1]
        t0 = self.sample_to_time(start_sampno)
        t1 = self.sample_to_time(start_sampno+nt)
        flags_t0 = self.meta_file.flags_at_time(t0)
        flags_t1 = self.meta_file.flags_at_time(t1)
        flags = flags_t0 | flags_t1
        if self.mask:
            for ibl, blid in enumerate(self.internal_baseline_order):
                a1,a2 = bl2ant(blid)
                ia1, ia2 = (a1 - 1), (a2 - 1)
                f = flags[ia1] or flags[ia2] 
                mask[ibl,...] = f
            
            dout_complex_data = np.ma.MaskedArray(data = dout_complex_data, mask = mask)

        return dout_complex_data

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
