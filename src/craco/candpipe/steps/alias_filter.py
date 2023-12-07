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
    parser.add_argument('--wcsfits', help='fitsfile for psf wcs info, useful for alias check', default=None)
 #   parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    return parser


class Step(ProcessingStep):

    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
        self.wcs_info = self.get_wcs()
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
        #from IPython import embed
        #embed()

        config = self.pipeline.config

        # # get the wcs info and start the alias filtering session
        # # 1. get the wcs info (read the arguments)
        # # 2. filtering current unknown sources (select all sources without a known pulsar/RACS crossmatch)
        # # 3. create a combined candidaets catalogue (8 possible alias location for each unknwon candidates)
        # # 4. run the crossmatch - using a combined racs/pulsar catalogue 
        # if self.pipeline.args.wcsfits is not None:
        
        # print(catalog_cross_match.Step(self.pipeline).angular_offset(20, 180, 20, 189))
        # radius, threshold = self.pipeline.config['filter_radius'], self.pipeline.config['threshold_alias']

        # get mean ra and dec from candidates file for further clustering 
        ra, dec = ind['ra_deg'].mean(), ind['dec_deg'].mean()

        # filetering catalogue
        catdf, catcoord = catalog_cross_match.Step(self.pipeline).filter_cat(ra=ra, 
                                        dec=dec, 
                                        catpath=config['catpath_alias'], 
                                        radius=config['filter_radius'], 
                                        racol='RA', 
                                        deccol='Dec')

        # create a new catalogue contains all unknown objects, each with 8 possible alias location
        alias_df = self.get_possible_alias_candidates(df=ind, )

        # run the crossmatch step!
        alias_df = catalog_cross_match.Step(self.pipeline).cross_matching(candidates=alias_df, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_alias'], 
                                       col_prefix='ALIAS', 
                                       key='Name')

        # save the candidates file
        outd = self.save_back_candfile(ind, alias_df)

        # outd = self.alias_filtering(ind, wcs_info, radius, threshold)
        
        # apply command line argument for minimum S/N and only return those values
        #if self.pipeline.args.cluster_min_sn is not None:
        #    outd = outd[outd['SNR'] > self.pipeline.args.cluster_min_sn]
        
        return outd


    def get_wcs(self):
        # read the wcs fitsfile 
        fitsfile = self.pipeline.args.wcsfits

        # Open the FITS file and get the header
        with fits.open(fitsfile) as hdul:
            wcs_info = WCS(hdul[0].header)

        return wcs_info


    # def get_source_coords(self, alias_lpix, alias_mpix, wcs_info):
    #     # Get the pixel values for all possible source locations and convert them to RA and Dec
    #     pixels = [((int(alias_lpix + wcs_info.array_shape[0])), int(alias_mpix)), 
    #             (int(alias_lpix - wcs_info.array_shape[0]), int(alias_mpix)),
    #             (int(alias_lpix), int(alias_mpix + wcs_info.array_shape[1])),
    #             (int(alias_lpix), int(alias_mpix - wcs_info.array_shape[1])),
    #             (int(alias_lpix + wcs_info.array_shape[0]), int(alias_mpix + wcs_info.array_shape[1])),
    #             (int(alias_lpix - wcs_info.array_shape[0]), int(alias_mpix - wcs_info.array_shape[1])),
    #             (int(alias_lpix + wcs_info.array_shape[0]), int(alias_mpix - wcs_info.array_shape[1])),
    #             (int(alias_lpix - wcs_info.array_shape[0]), int(alias_mpix + wcs_info.array_shape[1]))]

    #     coords = [wcs_info.pixel_to_world(*pixels[i]) for i in range(len(pixels))]
        
    #     return coords

    def get_source_coords(self, lpix, mpix):
        # Get the pixel values for all possible source locations and convert them to RA and Dec
        # make it works for a list of lpix/mpix
        xp, yp = self.wcs_info.array_shape[0], self.wcs_info.array_shape[1]
        lpix, mpix = np.array(lpix, dtype=np.int), np.array(mpix, dtype=np.int)

        lpixlist = list(lpix+xp) + list(lpix) + list(lpix-xp) + list(lpix+xp) + \
                     list(lpix-xp) + list(lpix+xp) + list(lpix) + list(lpix-xp)
        mpixlist = list(mpix+yp) + list(mpix+yp) + list(mpix+yp) + list(mpix) + \
                    list(mpix) + list(mpix-yp) + list(mpix-yp) + list(mpix-yp)
        
        return lpixlist, mpixlist


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
        lpixlist, mpixlist = self.get_source_coords(unknown_df['lpix'], unknown_df['mpix'])
        log.debug("obtained %s possible alias position", len(lpixlist))

        # get their skycoord
        coords = self.wcs_info.pixel_to_world(lpixlist, mpixlist)

        # create a new alias DataFrame 
        alias_df = pd.DataFrame()
        alias_df['ra_deg'] = coords.ra.deg
        alias_df['dec_deg'] = coords.dec.deg
        # alias_df['idx'] = list(unknown_df['idx']) * 8
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

        # find unique idx with possible alias 
        alias_df = alias_df[alias_idx]
        
        for idx, name, sep in zip(alias_df['idx'], alias_df[colname], alias_df[sepname]):
            df.loc[idx, colname] = name
            df.loc[idx, sepname] = sep

        return df





    # def filter_cat(self, alias_coord, radius):
    #     # Calculate the coordinates at a separation of radius degrees
    #     coord1 = alias_coord.directional_offset_by(0*u.degree, radius*u.degree)
    #     coord2 = alias_coord.directional_offset_by(90*u.degree, radius*u.degree)
    #     coord3 = alias_coord.directional_offset_by(180*u.degree, radius*u.degree)
    #     coord4 = alias_coord.directional_offset_by(270*u.degree, radius*u.degree)
        
    #     # Define the maximum and minimum RA and Dec values from coordinates
    #     max_ra = coord2.ra.deg
    #     min_ra = coord4.ra.deg
    #     max_dec = coord1.dec.deg
    #     min_dec = coord3.dec.deg
        
    #     # Load the source catalogue dataframe
    #     df_source = pd.read_csv(self.pipeline.config["catpath_alias"])

    #     # Perform boolean indexing
    #     if (max_ra > min_ra) and (abs(max_dec - min_dec) > 2):
    #         filtered_df_source = df_source[((df_source['RA'] <= max_ra) & (df_source['RA'] >= min_ra)) 
    #                                 & (df_source['Dec'] <= max_dec) & (df_source['Dec'] >= min_dec)]
    #     elif (max_ra < min_ra) and (abs(max_dec - min_dec) > 2):
    #         filtered_df_source = df_source[((df_source['RA'] <= max_ra) | (df_source['RA'] >= min_ra)) 
    #                                     & (df_source['Dec'] <= max_dec) & (df_source['Dec'] >= min_dec)]
    #     elif (max_ra > min_ra) and (abs(max_dec - min_dec) < 2):
    #         filtered_df_source = df_source[((df_source['RA'] <= max_ra) & (df_source['RA'] >= min_ra))]
    #     else:
    #         filtered_df_source = df_source[((df_source['RA'] <= max_ra) | (df_source['RA'] >= min_ra))]
        
    #     # Get the SkyCoords of the filtered source catalogue
    #     ref_coord = SkyCoord(ra=filtered_df_source['RA'].values, dec=filtered_df_source['Dec'].values, unit=u.degree)
        
    #     return filtered_df_source, ref_coord


    # def alias_filtering(self, df, wcs_info, radius, threshold):
    #     # Read the candidate file and get the candidates and their lpix and mpix values, 
    #     # and filter out all candidates that are not possible aliases
    #     df["Full Index"] = range(1, len(df)+1)
    #     df['Alias'] = None
    #     df['Alias_source'] = None
    #     df['Alias_sep'] = None
        
    #     alias_df = df[(df['SNR'] > 8) & (df['PSR_name'].isna()) & (df['RACS_name'].isna()) & (df['NEW_name'].isna())]

    #     # Loop through each possible alias in the candidate file
    #     for i in range(len(alias_df)):
            
    #         # Get the SkyCoord of the alias
    #         alias_lpix, alias_mpix = alias_df.iloc[i]['lpix'], alias_df.iloc[i]['mpix']
    #         alias_coord = wcs_info.pixel_to_world(alias_lpix, alias_mpix)

    #         # Get the SkyCoords of all possible source locations
    #         coords = self.get_source_coords(alias_lpix, alias_mpix, wcs_info)

    #         # Get the SkyCoords of all sources from the filtered source catalogue
    #         filtered_df_source, ref_coord = self.filter_cat(alias_coord, radius)

    #         # Perform crossmatching
    #         idx, sep, _ = zip(*[coords[i].match_to_catalog_sky(ref_coord) for i in range(len(coords))])
            
    #         # Get the minimum separation and the index of the source with the minimum separation
    #         min_sep = min(sep)
    #         min_idx = idx[np.argmin(sep)]

    #         # Print the results of the crossmatching and the possible source of the alias
    #         if min_sep.arcsec <= threshold:
    #             log.info(f"Alias at {(alias_lpix, alias_mpix)}. Possible source of aliasing: {filtered_df_source.iloc[min_idx]['Name']} at RA: {filtered_df_source.iloc[min_idx]['RA']}, Dec: {filtered_df_source.iloc[min_idx]['Dec']} and a separation of {float(min_sep.arcsec):.5f} arcsec")
    #             df.loc[df['Full Index'] == alias_df.iloc[i]['Full Index'], 'Alias'] = True
    #             df.loc[df['Full Index'] == alias_df.iloc[i]['Full Index'], 'Alias_source'] = filtered_df_source.iloc[min_idx]['Name']
    #             df.loc[df['Full Index'] == alias_df.iloc[i]['Full Index'], 'Alias_sep'] = min_sep.arcsec
    #         else:
    #             print(f"No source found within {threshold} arcsec of alias")
    #             df.loc[df['Full Index'] == alias_df.iloc[i]['Full Index'], 'Alias'] = False

    #     return df

    

    def close(self):
        pass
