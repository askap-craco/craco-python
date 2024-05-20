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
from astropy.coordinates import SkyCoord
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
        
        # select catalogue objects located within the observation field of view 
        for i, catpath in enumerate(config['catpath']):
            log.debug('Selecting sources from existing catalogue %s', catpath)
            filter_radius = config['filter_radius']
            catdf, catcoord = self.filter_cat(ra=ra, 
                                              dec=dec, 
                                              catpath=catpath, 
                                              radius=filter_radius, 
                                              racol=config['catcols']['ra'][i], 
                                              deccol=config['catcols']['dec'][i])

            log.debug('Found %s sources within %s degree radius in %s', len(catdf), config['filter_radius'], catpath)
            if len(catdf) == 0:
                warnings.warn('Catalog {catpath} contains no sources within {filter_radius} of ({ra},{dec})')    

            log.debug('Starting in-field sources crossmatch %s', catpath)

            outd = self.cross_matching(candidates=ind, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_crossmatch'][i], 
                                       col_prefix=config['catcols']['output_prefix'][i], 
                                       key=config['catcols']['input_colname'][i])

            log.debug('Crossmatch finished for %s', catpath)
            
            ind = outd    
        
        return outd


    def angular_offset(self, ra1, dec1, ra2, dec2):
        # in unit of degree
        # just don't use stupid astropy separation - that's too slow!! 

        phi1 = ra1 * np.pi / 180
        phi2 = ra2 * np.pi / 180
        theta1 = dec1 * np.pi / 180
        theta2 = dec2 * np.pi / 180

        cos_sep_radian = np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1-phi2)
        sep = np.arccos(cos_sep_radian) * 180 / np.pi

        return sep

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
            catcoord = SkyCoord(catra[select_bool], catdec[select_bool], unit=(units.degree))
            d = catdf.iloc[select_bool], catcoord

            # cache the answer if the input was reasonable            
            # ra will be Nan if the input block was empty.
            # if you cache it it will break forever. This would suck.
            if not np.isnan(ra):
                self.catalogs[catpath] = d
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
            for i, catpath in enumerate(config['catpath']):
                log.debug('Selecting sources from existing catalogue %s', catpath)
                filter_radius = config['filter_radius']
                catdf, catcoord = self.filter_cat(ra=ra, 
                                                dec=dec, 
                                                catpath=catpath, 
                                                radius=filter_radius, 
                                                racol=config['catcols']['ra'][i], 
                                                deccol=config['catcols']['dec'][i])


    def cross_matching(self, candidates, catalogue, coord, 
                       threshold=30, 
                       col_prefix='PSR', key='PSRJ'):
        '''
        @author: Yu Wing Joshua Lee
        Threshold is in arcsecond.
        Catalogue should be in csv format.
        '''
        threshold = threshold * units.arcsec
        
        # add column to candidate list and save as a new file
        colname = col_prefix + '_name'
        sepname = col_prefix + '_sep'

        if len(catalogue) == 0:
            candidates[colname] = [None] * len(candidates)
            candidates[sepname] = [None] * len(candidates)
            return candidates

        cand_radec = SkyCoord(ra=candidates['ra_deg'], dec=candidates['dec_deg'], unit=(units.degree, units.degree), frame='icrs')

        idx, sep2d, sep3d = cand_radec.match_to_catalog_sky(coord)
        combined = [[idx, sep2d.arcsec] if sep2d<threshold else[None, None] for idx, sep2d in zip(idx, sep2d)]

        colname_list = []
        sepname_list = []

        for index, pair in enumerate(combined):
            pulsar_name, pulsar_distance = pair
            colname_list.append(catalogue[key].iloc[pulsar_name] if pulsar_name is not None else None)
            sepname_list.append(pulsar_distance)

        candidates[colname] = colname_list
        candidates[sepname] = sepname_list
        
        return candidates


    def close(self):
        pass


    