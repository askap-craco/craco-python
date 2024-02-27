#!/usr/bin/env python
"""
Clusters the input

Copyright (C) CSIRO 2023
"""
import pylab
import numpy as np
import os
import sys
import logging
from craco.candpipe.candpipe import ProcessingStep

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd

from craco.candpipe.steps import catalog_cross_match


log = logging.getLogger(__name__)

__author__ = '''Akhil Jaini <ajaini@swin.edu.au>; 
                Yuanming Wang <yuanmingwang@swin.edu.au>'''


def get_parser():
    '''
    Create and return an argparse.ArgumentParser with the arguments you need on the commandline
    Make sure add_help=False
    '''
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
 #   parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    return parser


class Step(ProcessingStep):

    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
        self.pipeline.wcs_info = self.get_wcs()
        # You might want to use some of these attributes

        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s psf=%s arguments=%s',
                 p.srcdir, p.beamno, p.cand_fname, p.uvfits_fname,
                 p.cas_fname, p.ics_fname, p.pcb_fname, p.psf_fname, p.args)
                  

    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate dataframe and returns a new dataframe of classified data
        Hopefully with fewer, better quality candidtes
        '''

        log.debug('Got %d candidates type=%s, columns=%s', len(ind), type(ind), ind.columns)
        #from IPython import embed
        #embed()

        p = self.pipeline
        config = self.pipeline.config

        ## get the wcs info and start the alias filtering session
        ## 1. get the wcs info (read the arguments)
        ## 2. filtering current unknown sources (select all sources without a known pulsar/RACS crossmatch)
        ## 3. create a combined candidaets catalogue (8 possible alias location for each unknwon candidates)
        ## 4. run the crossmatch - using a combined racs/pulsar catalogue 
    
        # get mean ra and dec from candidates file for further clustering 
        ra, dec = ind['ra_deg'].mean(), ind['dec_deg'].mean()

        # filetering catalogue
        catdf, catcoord = catalog_cross_match.Step(p).filter_cat(ra=ra, 
                                        dec=dec, 
                                        catpath=config['catpath_alias'], 
                                        radius=config['filter_radius'], 
                                        racol='RA', 
                                        deccol='Dec')

        # create a new catalogue contains all unknown objects, each with 8 possible alias location
        alias_df = self.get_possible_alias_candidates(df=ind, )

        # run the crossmatch step!
        alias_df = catalog_cross_match.Step(p).cross_matching(candidates=alias_df, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_alias'], 
                                       col_prefix='ALIAS', 
                                       key='Name')

        # save the candidates file
        outd = self.save_back_candfile(ind, alias_df)
        
        # apply command line argument for minimum S/N and only return those values
        #if self.pipeline.args.cluster_min_sn is not None:
        #    outd = outd[outd['SNR'] > self.pipeline.args.cluster_min_sn]
        
        return outd


    def get_wcs(self):
        # read the wcs fitsfile 
        fitsfile = self.pipeline.psf_fname

        # Open the FITS file and get the header
        with fits.open(fitsfile) as hdul:
            wcs_info = WCS(hdul[0].header)

        return wcs_info


    def get_source_coords(self, ra, dec):
        # Get FoV (in degree)
        # make it works for a list of ra and dec 
        ra_fov = np.abs(self.pipeline.psf_header['NAXIS1'] * self.pipeline.psf_header['CDELT1'])
        dec_fov = np.abs(self.pipeline.psf_header['NAXIS2'] * self.pipeline.psf_header['CDELT2'])

        ra, dec = np.array(ra, dtype=float), np.array(dec, dtype=float)

        xrr = ra_fov / np.cos(dec/180*np.pi)
        yrr = dec_fov

        ra_alias = list(ra+xrr) + list(ra) + list(ra-xrr) + list(ra+xrr) + \
                     list(ra-xrr) + list(ra+xrr) + list(ra) + list(ra-xrr)
        dec_alias = list(dec+yrr) + list(dec+yrr) + list(dec+yrr) + list(dec) + \
                    list(dec) + list(dec-yrr) + list(dec-yrr) + list(dec-yrr)
        
        return ra_alias, dec_alias


    def get_possible_alias_candidates(self, df):
        # only select unknown objects 
        # i.e., output_prefix has a nan value 
        '''
        df: input candidates catalogue 
        '''
        # df['idx'] = range(len(df))
        unknown_idx = (df['PSR_name'].isna()) & (df['RACS_name'].isna()) & (df['NEW_name'].isna())
        unknown_df = df[unknown_idx]
        log.debug("%s candidates do not have cross-matched sources - continue to alias filtering...", sum(unknown_idx))
        
        # calculate their alias location 
        ra_alias, dec_alias = self.get_source_coords(unknown_df['ra_deg'], unknown_df['dec_deg'])
        log.debug("obtained %s possible alias position", len(ra_alias))

        # create a new alias DataFrame 
        alias_df = pd.DataFrame()
        alias_df['ra_deg'] = ra_alias
        alias_df['dec_deg'] = dec_alias
        alias_df['idx'] = list(unknown_df.index) * 8

        return alias_df


    def save_back_candfile(self, df, alias_df):
        '''
        df: original candidates table
        alias_df: possible alias table 
        '''
        alias_idx = (~alias_df['ALIAS_name'].isna())

        colname = 'ALIAS_name'
        sepname = 'ALIAS_sep'

        df[colname] = [None] * len(df)
        df[sepname] = [None] * len(df)

        # if there's no alias at all...
        if alias_idx.sum() == 0:
            log.debug("No possible alias sources")
            return df

        log.info("Find %s possible aliasing sources", alias_idx.sum())

        # find unique idx with possible alias 
        alias_df = alias_df[alias_idx]
        
        for idx, name, sep in zip(alias_df['idx'], alias_df[colname], alias_df[sepname]):
            df.loc[idx, colname] = name
            df.loc[idx, sepname] = sep

        return df



    def close(self):
        pass
