#!/usr/bin/env python
"""
Clusters the input

Copyright (C) CSIRO 2023
"""
import pylab
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import warnings

from astropy import wcs
from astropy.io import fits
from craco.candpipe.candpipe import ProcessingStep
# from astropy.coordinates import SkyCoord
from astropy import units

log = logging.getLogger(__name__)

__author__ = '''Akhil Jaini <ajaini@swin.edu.au>; 
                Yu Wing Joshua Lee <ylee2156@uni.sydney.edu.au>; 
                Yuanming Wang <yuanmingwang@swin.edu.au>; 
                Ziteng Wang <ztwang201605@gmail.com>'''


def get_parser():
    '''
    Create and return an argparse.ArgumentParser with the arguments you need on the commandline
    Make sure add_help=False
    '''
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    # parser.add_argument('--min_flux', type=float, help='Minimum signal flux for querying the catalogue', default=None)
    # parser.add_argument('--ra_err', type=float, help='Maximum error to be allowed in RA measurement for querying the catalogue', default=None)
    # parser.add_argument('--dec_err', type=float, help='Maximum error to be allowed in DEC measurement for querying the catalogue', default=None)
    return parser


class Step(ProcessingStep):
    
    step_name = "cross_match"
    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
        self.catalogs = {}
        config = self.pipeline.config
        
        # TODO: To pre-cache catalogs we need the FOV pointing ra/dec from the WCS.
        
        # You might want to use some of these attributes

        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
                 p.srcdir, p.beamno, p.cand_fname, p.uvfits_fname,
                 p.cas_fname, p.ics_fname, p.pcb_fname, p.args)
                  

    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate dataframe and returns a new dataframe of classified data
        Hopefully with fewer, better quality candidtes
        '''

        log.debug('Got %d candidates type=%s, columns=%s', len(ind), type(ind), ind.columns)
        config = self.pipeline.config 

        # get median ra and dec from candidates file for further clustering 
        ra, dec = ind['ra_deg'].median(), ind['dec_deg'].median()
        log.debug('Find median coordinates for this field is %s %s', ra, dec)

        outd = self.classify_cands(ind, ra, dec)
        
        return outd


    def angular_offset(self, ra1, dec1, ra2, dec2):
        # in unit of degree
        # just don't use stupid astropy separation - that's too slow!! 
        phi1 = ra1 * np.pi / 180
        phi2 = ra2 * np.pi / 180
        theta1 = dec1 * np.pi / 180
        theta2 = dec2 * np.pi / 180

        cos_sep_radian = np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1-phi2)
        # Clip values to the valid range for arccos to avoid numerical issues
        cos_sep_radian = np.clip(cos_sep_radian, -1.0, 1.0) 
        sep = np.arccos(cos_sep_radian) * 180 / np.pi

        return sep # unit of deg

    def match_coords(self, ra1, dec1, ra2, dec2):
        # in unit of degree
        # works for a list of coord1 and coord2, they should be numpy array 
        # return the closest coord2 for each coord1 
        # replace the astropy match_to_catalog_sky function 
        ra1, ra2 = np.meshgrid(ra1, ra2, indexing='ij')
        dec1, dec2 = np.meshgrid(dec1, dec2, indexing='ij')

        ra1 = ra1.astype(np.float64)
        dec1 = dec1.astype(np.float64)
        ra2 = ra2.astype(np.float64)
        dec2 = dec2.astype(np.float64)

        sep = self.angular_offset(ra1, dec1, ra2, dec2)
        
        min_sep_idx = np.argmin(sep, axis=1)
        min_sep = np.min(sep, axis=1) * 3600 # unit of arcsec
        return min_sep_idx, min_sep

    def load_catalog(self, catpath, racol, deccol):
        '''
        ### load catalog here - assume it is csv
        caches in the catalogs dictionary
        catdf = pd.read_csv(catpath)
        catra, catdec = np.array(catdf[racol]), np.array(catdf[deccol])
        '''
        catdf = pd.read_csv(catpath)
        catra, cadec = np.array(catdf[racol]), np.array(catdf[deccol])
        d = (catdf, catra, cadec)

        return d


    def filter_cat(self, ra, dec, catpath, radius=2, racol="RA", deccol="Dec"):
        """
        filter catalogues based on the radius and central ghost

        params:
        ---------
        ra, dec: float
            central coordinates in degrees
        radius: float, int
            in degrees
        """
        if catpath not in self.catalogs.keys():
            catdf, catra, catdec = self.load_catalog(catpath, racol, deccol)
            sep = self.angular_offset(ra, dec, catra, catdec)
            select_bool = sep < radius
            # catcoord = SkyCoord(catra[select_bool], catdec[select_bool], unit=(units.degree))
            catcoord = [catra[select_bool], catdec[select_bool]]
            d = catdf.iloc[select_bool], catcoord

            # cache the answer if the input was reasonable            
            # ra will be Nan if the input block was empty.
            # if you cache it it will break forever. This would suck.
            if not np.isnan(ra):
                self.catalogs[catpath] = d
                log.info('Loaded %d sources from %s within %f of (%f, %f)', sum(select_bool), catpath, radius, ra, dec)
                     
        else:        
            d = self.catalogs[catpath]

        return d

    def set_current_wcs(self, wcs, iblk):
        '''
        If catalogs not loaded, load them using WCS phase center
        '''
        config = self.pipeline.config
        ra, dec = self.pipeline.get_current_phase_center()
        if len(self.catalogs) == 0:
            log.info('Loading %d catalogs for (%f,%f)', len(config['catpath']), ra, dec)
            for i, catpath in enumerate(config['catpath']):
                log.debug('Selecting sources from existing catalogue %s', catpath)
                filter_radius = config['filter_radius']
                catdf, catcoord = self.filter_cat(ra=ra, 
                                                dec=dec, 
                                                catpath=catpath, 
                                                radius=filter_radius, 
                                                racol=config['catcols']['ra'][i], 
                                                deccol=config['catcols']['dec'][i])


    def cross_matching(self, candidates, catalogue, coord, threshold=30, key='PSRJ'):
        '''
        Threshold is in arcsecond.
        Catalogue should be in csv format.
        '''
        # add column to candidate list and save as a new file
        col_prefix = 'MATCH'
        colname = col_prefix + '_name'
        sepname = col_prefix + '_sep'

        if len(catalogue) == 0:
            condition = np.full((len(candidates),), False, dtype=bool)
            return candidates, condition

        # astropy version -> super slow 
        # cand_radec = SkyCoord(ra=candidates['ra_deg'], dec=candidates['dec_deg'], unit=(units.degree, units.degree), frame='icrs')
        # idx, sep2d, sep3d = cand_radec.match_to_catalog_sky(coord)
        # sep2d = sep2d.arcsec

        # self-defined crossmatch -> works fine in most of cases, sometimes gives a slightly different result to astropy 
        idx, sep2d = self.match_coords(candidates['ra_deg'].values, candidates['dec_deg'].values, coord[0], coord[1])
        condition = sep2d < threshold 

        candidates.loc[condition, colname] = catalogue[key].iloc[idx].values[condition]
        candidates.loc[condition, sepname] = sep2d[condition]
        
        return candidates, condition 


    def classify_cands(self, candidates, ra, dec, ):
        '''
        LABEL
            1. PSR 
            2. RACS
            3. CUSTOM
            4. UNKNOWN 
        ALIAS:
            True
            False 
        '''
        config = self.pipeline.config 
        conditions = []
        log.info('Total %s candidates waiting for crossmatch..', len(candidates))

        for i, catpath in enumerate(config['catpath']):
            filter_radius = config['filter_radius']
            catdf, catcoord = self.filter_cat(ra=ra, 
                                              dec=dec, 
                                              catpath=catpath, 
                                              radius=filter_radius, 
                                              racol=config['catcols']['ra'][i], 
                                              deccol=config['catcols']['dec'][i])

            log.debug('Found %s sources within %s degree radius in %s', len(catdf), config['filter_radius'], catpath)
            candidates, condition = self.cross_matching(candidates=candidates, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_crossmatch'][i], 
                                       key=config['catcols']['input_colname'][i])

            conditions.append(condition)
        
        classes = config['catcols']['output_prefix']
        candidates.loc[:, 'LABEL'] = np.select(conditions, classes, default='UNKNOWN')
        candidates.loc[:, 'ALIASED'] = [0] * len(candidates)

        return candidates


    def close(self):
        pass


    