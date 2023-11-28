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
    parser.add_argument('--min_flux', type=float, help='Minimum signal flux for querying the catalogue', default=None)
    parser.add_argument('--ra_err', type=float, help='Maximum error to be allowed in RA measurement for querying the catalogue', default=None)
    parser.add_argument('--dec_err', type=float, help='Maximum error to be allowed in DEC measurement for querying the catalogue', default=None)
    return parser


class Step(ProcessingStep):
    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
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

        # get mean ra and dec from candidates file for further clustering 
        ra, dec = ind['ra_deg'].mean(), ind['dec_deg'].mean()
        
        # select catalogue objects located within the observation field of view 
        for i, catpath in enumerate(config['catpath']):
            log.debug('Crossmatching with existing catalogue %s', catpath)
            catdf, catcoord = self.filter_cat(ra=ra, 
                                              dec=dec, 
                                              catpath=catpath, 
                                              radius=config['filter_radius'], 
                                              racol=config['catcols']['ra'][i], 
                                              deccol=config['catcols']['dec'][i])

            log.debug('Filtering in-field sources finished %s', catpath)

            outd = self.cross_matching(candidates=ind, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_crossmatch'][i], 
                                       col_prefix=config['catcols']['output_prefix'][i], 
                                       key=config['catcols']['input_colname'][i])

            log.debug('Crossmatch finished for %s', catpath)
            
            ind = outd    
        
        return outd
    

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
        # # later on take catalogue from buffer?
        # ctrcoord = SkyCoord(ra, dec, unit=units.degree)

        # ### load catalog here - assume it is csv
        # catdf = pd.read_csv(catpath)
        # catcoord = SkyCoord(catdf[racol], catdf[deccol], unit=units.degree) # this step is really slow 
        # sep = ctrcoord.separation(catcoord)

        # select_bool = sep.value < radius

        # load the catalog here
        catdf = pd.read_csv(catpath)
        catra, catdec = catdf[racol], catdf[deccol]

        # separation - to improve the speed, so did a improper selection
        rasep, decsep = np.abs(catra-ra), np.abs(catdec-dec)
        select_bool = np.array(rasep < radius) & np.array(decsep < radius)
        catcoord = SkyCoord(catdf.iloc[select_bool][racol], catdf.iloc[select_bool][deccol], unit=units.degree)
        
        return catdf.iloc[select_bool], catcoord


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


    