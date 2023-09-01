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
from psrqpy import QueryATNF
from astropy import wcs
from astropy.io import fits
from craco.candpipe.candpipe import ProcessingStep
from astropy.coordinates import SkyCoord
from astropy import units

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au> and others put your name here"


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
 #   parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    return parser

class Step(ProcessingStep):
    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
        # You might want to use some of these attributes

#        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
#                  p.srcdir, p.beamno, p.cand_fname, p.uvfits_fname,
#                  p.cas_fname, p.ics_fname, p.pcb_fname, p.args)
                  

    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate dataframe and returns a new dataframe of classified data
        Hopefully with fewer, better quality candidtes
        '''

        #log.debug('Got %d candidates type=%s, columns=%s', len(ind), type(ind), ind.columns)
        #from IPython import embed
        #embed()

        outd = ind
        
        # apply command line argument for minimum S/N and only return those values
        #if self.pipeline.args.cluster_min_sn is not None:
        #    outd = outd[outd['SNR'] > self.pipeline.args.cluster_min_sn]
        

        return outd
    
    # Function to query the ATNF Pulsar Catalogue: https://www.atnf.csiro.au/research/pulsar/psrcat/
    # Taking in arguments of flux, ra_err and dec_err to give threshold values for pulsar filtering from the catalogue  
    def QueryPSRCat(self, flux=0.03,ra_err=1e-1,dec_err=1e-1):
        if os.path.exists('psrcat_Main.csv'):
            print("Catalogue file already exists")
        else:
            query = QueryATNF()
            query.query_params = ['JName', 'RaJD', 'DecJD', 'P0', 'DM', 'W50', 'W10', 'S400', 'S1400']

            # Converting the query into a Pandas dataframe
            df = query.pandas

            # Only having pulsars that have RA and DEC measured above a certain astrometric precision, and whose error values are provided
            df['RAJD_ERR'].replace('', np.nan, inplace=True)
            df.dropna(subset=['RAJD_ERR'], inplace=True)
            df = df.drop(df[df['RAJD_ERR'] > ra_err].index)
            df['DECJD_ERR'].replace('', np.nan, inplace=True)
            df.dropna(subset=['DECJD_ERR'], inplace=True)
            df = df.drop(df[df['DECJD_ERR'] > dec_err].index)
            
            # Only having pulsars above a certain threshold and dropping the remaining ones
            df['C1'] = df.apply(lambda x: x['S1400']*x['P0']/x['W50'], axis=1)
            df['C1'].replace('', np.nan, inplace=True)
            df.dropna(subset=['C1'], inplace=True)
            df = df.drop(df[df['C1'] < flux].index)
            df = df.sort_values(by=['C1'], ascending=False)
            df = df[['JNAME', 'RAJD', 'RAJD_ERR', 'DECJD', 'DECJD_ERR', 'P0', 'P0_ERR', 'DM', 'DM_ERR', 
                    'W50', 'W50_ERR', 'W10', 'W10_ERR', 'S400', 'S400_ERR', 'S1400', 'S1400_ERR', 'C1']]            # Rearranging the columns
            
            numstring = 'Version {} of the ATNF catalogue contains {} pulsars, and after sorting contains {} pulsars.'
            print(numstring.format(query.get_version, query.num_pulsars, df.shape[0]))                              # For testing only, can be removed
            df.to_csv('psrcat_Main.csv')


    # Function to iterate through all of the FITS header files (currently only works with files that are already in the directory, 
    # but a realtime version can be implemented to read the files as they are dumped during observation). The function then reads
    # the RA, DEC, lpix and mpix values to find the field of view of the corresponding candidate file. Finally, the function 
    # compares this with the sources in the catalogue and returns two dataframes (also saved as CSV files): one with the sources 
    # in the field of view, another with the sources out of the field of view. The next step of the pipeline will have both these 
    # files as input

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
        # later on take catalogue from buffer?
        ctrcoord = SkyCoord(ra, dec, unit=units.degree)

        ### load catalog here - assume it is csv
        catdf = pd.read_csv(catpath)
        catcoord = SkyCoord(catdf[racol], catdf[deccol], unit=units.degree)
        sep = ctrcoord.separation(catcoord)

        select_bool = sep.value < radius

        return catdf.iloc[select_bool], catcoord[select_bool]


    def check_field(dirpath=os.curdir):                                                                         # Directory path of FITS files to be provided
        fitsfiles = [f for f in os.listdir(dirpath) if f.endswith(".fits")]
        
        # Looping through the FITS files to get required info from the headers
        for i in range(len(fitsfiles)):
            header = fits.getheader(fitsfiles[i])
            world_coords = wcs.WCS(header)
            l_axis = fits.getval(fitsfiles[i], 'NAXIS1')
            m_axis = fits.getval(fitsfiles[i], 'NAXIS2')
            ra_cen = fits.getval(fitsfiles[i], 'CRVAL1')
            dec_cen = fits.getval(fitsfiles[i], 'CRVAL2')
            
            # Finding the field of view of the candidate file
            [ra_primary1, dec_primary1], [ra_primary2, dec_primary2] = world_coords.wcs_pix2world([[0,0], [l_axis-1,m_axis-1]], 0)
            
            # Adding an excess that is twice the primary FoV to the searchable FoV to find the alised FoV, 
            # and ensuring that the RA and DEC are within their bounds
            excess_ra, excess_dec = abs(ra_primary1 - ra_primary2), abs(dec_primary1 - dec_primary2)
            ra_aliased_min = ((np.minimum(ra_primary1, ra_primary2) - excess_ra) % 360 + 360) % 360
            ra_aliased_max = ((np.maximum(ra_primary1, ra_primary2) + excess_ra) % 360 + 360) % 360
            dec_aliased_min = np.clip(np.minimum(dec_primary1, dec_primary2) - excess_dec, -90, 90)
            dec_aliased_max = np.clip(np.maximum(dec_primary1, dec_primary2) + excess_dec, -90, 90)
            
            # Calling the pulsar query catalogue and adding a column based on whether the pulsars are within the FoV 
            df = pd.read_csv('PSRCat_Main.csv')
            df['In_Field'] = df.apply(lambda x: 'YES' if ((ra_aliased_min <= x['RAJD'] <= ra_aliased_max) 
                                                        & (dec_aliased_min <= x['DECJD'] <= dec_aliased_max)) else 'NO', axis=1)
            
            # Creating two dataframes, one for sources within the FoV and another for sources outside
            df_InField = df.drop(df[df['In_Field'] == 'NO'].index)
            # df_InField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_InField.csv')
            df_NotInField = df.drop(df[df['In_Field'] == 'YES'].index)
            # df_NotInField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_NotInField.csv')
        
        return df_InField, df_NotInField

    def close(self):
        pass
