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
<<<<<<< HEAD
=======
# from psrqpy import QueryATNF
>>>>>>> d655d8fa553b1dc470934b768c74d102012ab559
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

        config = self.pipeline.config 

        # get mean ra and dec from candidates file for further clustering 
        ra, dec = ind['ra_deg'].mean(), ind['dec_deg'].mean()
        # select catalogue objects located within the observation field of view 
        for i, catpath in enumerate(config['catpath']):
            catdf, catcoord = self.filter_cat(ra=ra, 
                                              dec=dec, 
                                              catpath=catpath, 
                                              radius=config['filter_radius'], 
                                              racol=config['catcols']['ra'][i], 
                                              deccol=config['catcols']['dec'][i])

            outd = self.cross_matching(candidates=ind, 
                                       catalogue=catdf, 
                                       coord=catcoord, 
                                       threshold=config['threshold_crossmatch'][i], 
                                       col_prefix=config['catcols']['output_prefix'][i], 
                                       key=config['catcols']['input_colname'][i])
            
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
        # later on take catalogue from buffer?
        ctrcoord = SkyCoord(ra, dec, unit=units.degree)

        ### load catalog here - assume it is csv
        catdf = pd.read_csv(catpath)
        catcoord = SkyCoord(catdf[racol], catdf[deccol], unit=units.degree)
        sep = ctrcoord.separation(catcoord)

        select_bool = sep.value < radius

        return catdf.iloc[select_bool], catcoord[select_bool]

    def cross_matching(self, candidates, catalogue, coord, 
                       threshold=30, 
                       col_prefix='PSR', key='PSRJ'):
        '''
        @author: Yu Wing Joshua Lee
        Important: upgrade numpy(1.25.2) and astropy(5.3.2) before running.
        Input_file and output_file are the path to the candidate list file.
        Threshold is in arcsecond.
        Catalogue should be in csv format.
        '''
        #Check numpy and astropy version
        # numpy_required_version = "1.25.2"
        # astropy_required_version = "5.3.2"
        # if astropy.__version__ < astropy_required_version or np.__version__ < numpy_required_version:
        #     raise ImportError(f"Astropy version {astropy_required_version} and Numpy version {numpy_required_version} or higher are required.")
        
        #Read candidate list and adjust threshold unit
        # df = pd.read_csv(input_file, index_col=None, sep='\t+',engine='python')
        # candidates = df.drop(df.index[-1])
        # threshold = threshold/3600
        threshold = threshold * units.arcsec
        
        #Read from psrcat if no catalogue is supplied. Generate a catalogue of pulsar around the beam center.
        #else convert the catalogue into pandas data frame.
        # if catalogue == None:
        #     ra_mean = candidates["ra_deg"].mean()
        #     dec_mean = candidates["dec_deg"].mean()
        #     limit = "RAJD > " + str(ra_mean - 5) + "&& RAJD < " + str(ra_mean + 5) + "&& DECJD > " +str(dec_mean - 5) +"&& DECJD < " + str(dec_mean + 5)
        #     query = QueryATNF(params=["NAME", "RAJD", "DECJD"], condition=limit)
        #     pulsar_list = query.pandas
        # else:
            # pulsar_list = pd.read_csv(catalogue)
        
        #Cross-checking using astropy match_to_catalog_sky

        #add column to candidate list and save as a new file
        colname = col_prefix + '_name'
        sepname = col_prefix + '_sep'
        new_columns = {colname:[], sepname:[]}

        if len(catalogue) == 0:
            new_df = pd.DataFrame(new_columns)
            result_df = pd.concat([candidates, new_df],axis=1)
            return result_df

        cand_radec = SkyCoord(ra=candidates['ra_deg'], dec=candidates['dec_deg'], unit=(units.degree, units.degree), frame='icrs')
        idx, sep2d, sep3d = cand_radec.match_to_catalog_sky(coord)
        combined = [[idx, sep2d.arcsec] if sep2d<threshold else[None, None] for idx, sep2d in zip(idx, sep2d)]

        for index, pair in enumerate(combined):
            pulsar_name, pulsar_distance = pair
            new_columns[colname].append(catalogue[key].iloc[pulsar_name] if pulsar_name is not None else None)
            new_columns[sepname].append(pulsar_distance)
        new_df = pd.DataFrame(new_columns)
        result_df = pd.concat([candidates, new_df],axis=1)
        
        return result_df


    def close(self):
        pass


    # # Function to query the ATNF Pulsar Catalogue: https://www.atnf.csiro.au/research/pulsar/psrcat/
    # # Taking in arguments of flux, ra_err and dec_err to give threshold values for pulsar filtering from the catalogue  
    # def QueryPSRCat(self, flux=0.03,ra_err=1e-1,dec_err=1e-1):
    #     if os.path.exists('psrcat_Main.csv'):
    #         print("Catalogue file already exists")
    #     else:
    #         query = QueryATNF()
    #         query.query_params = ['JName', 'RaJD', 'DecJD', 'P0', 'DM', 'W50', 'W10', 'S400', 'S1400']

    #         # Converting the query into a Pandas dataframe
    #         df = query.pandas

    #         # Only having pulsars that have RA and DEC measured above a certain astrometric precision, and whose error values are provided
    #         df['RAJD_ERR'].replace('', np.nan, inplace=True)
    #         df.dropna(subset=['RAJD_ERR'], inplace=True)
    #         df = df.drop(df[df['RAJD_ERR'] > ra_err].index)
    #         df['DECJD_ERR'].replace('', np.nan, inplace=True)
    #         df.dropna(subset=['DECJD_ERR'], inplace=True)
    #         df = df.drop(df[df['DECJD_ERR'] > dec_err].index)
            
    #         # Only having pulsars above a certain threshold and dropping the remaining ones
    #         df['C1'] = df.apply(lambda x: x['S1400']*x['P0']/x['W50'], axis=1)
    #         df['C1'].replace('', np.nan, inplace=True)
    #         df.dropna(subset=['C1'], inplace=True)
    #         df = df.drop(df[df['C1'] < flux].index)
    #         df = df.sort_values(by=['C1'], ascending=False)
    #         df = df[['JNAME', 'RAJD', 'RAJD_ERR', 'DECJD', 'DECJD_ERR', 'P0', 'P0_ERR', 'DM', 'DM_ERR', 
    #                 'W50', 'W50_ERR', 'W10', 'W10_ERR', 'S400', 'S400_ERR', 'S1400', 'S1400_ERR', 'C1']]            # Rearranging the columns
            
    #         numstring = 'Version {} of the ATNF catalogue contains {} pulsars, and after sorting contains {} pulsars.'
    #         print(numstring.format(query.get_version, query.num_pulsars, df.shape[0]))                              # For testing only, can be removed
    #         df.to_csv('psrcat_Main.csv')


    # Function to iterate through all of the FITS header files (currently only works with files that are already in the directory, 
    # but a realtime version can be implemented to read the files as they are dumped during observation). The function then reads
    # the RA, DEC, lpix and mpix values to find the field of view of the corresponding candidate file. Finally, the function 
    # compares this with the sources in the catalogue and returns two dataframes (also saved as CSV files): one with the sources 
    # in the field of view, another with the sources out of the field of view. The next step of the pipeline will have both these 
    # files as input


    # def check_field(dirpath=os.curdir):                                                                         # Directory path of FITS files to be provided
    #     fitsfiles = [f for f in os.listdir(dirpath) if f.endswith(".fits")]
        
    #     # Looping through the FITS files to get required info from the headers
    #     for i in range(len(fitsfiles)):
    #         header = fits.getheader(fitsfiles[i])
    #         world_coords = wcs.WCS(header)
    #         l_axis = fits.getval(fitsfiles[i], 'NAXIS1')
    #         m_axis = fits.getval(fitsfiles[i], 'NAXIS2')
    #         ra_cen = fits.getval(fitsfiles[i], 'CRVAL1')
    #         dec_cen = fits.getval(fitsfiles[i], 'CRVAL2')
            
    #         # Finding the field of view of the candidate file
    #         [ra_primary1, dec_primary1], [ra_primary2, dec_primary2] = world_coords.wcs_pix2world([[0,0], [l_axis-1,m_axis-1]], 0)
            
    #         # Adding an excess that is twice the primary FoV to the searchable FoV to find the alised FoV, 
    #         # and ensuring that the RA and DEC are within their bounds
    #         excess_ra, excess_dec = abs(ra_primary1 - ra_primary2), abs(dec_primary1 - dec_primary2)
    #         ra_aliased_min = ((np.minimum(ra_primary1, ra_primary2) - excess_ra) % 360 + 360) % 360
    #         ra_aliased_max = ((np.maximum(ra_primary1, ra_primary2) + excess_ra) % 360 + 360) % 360
    #         dec_aliased_min = np.clip(np.minimum(dec_primary1, dec_primary2) - excess_dec, -90, 90)
    #         dec_aliased_max = np.clip(np.maximum(dec_primary1, dec_primary2) + excess_dec, -90, 90)
            
    #         # Calling the pulsar query catalogue and adding a column based on whether the pulsars are within the FoV 
    #         df = pd.read_csv('PSRCat_Main.csv')
    #         df['In_Field'] = df.apply(lambda x: 'YES' if ((ra_aliased_min <= x['RAJD'] <= ra_aliased_max) 
    #                                                     & (dec_aliased_min <= x['DECJD'] <= dec_aliased_max)) else 'NO', axis=1)
            
    #         # Creating two dataframes, one for sources within the FoV and another for sources outside
    #         df_InField = df.drop(df[df['In_Field'] == 'NO'].index)
    #         # df_InField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_InField.csv')
    #         df_NotInField = df.drop(df[df['In_Field'] == 'YES'].index)
    #         # df_NotInField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_NotInField.csv')
        
    #     return df_InField, df_NotInField