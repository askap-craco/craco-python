#!/usr/bin/env python
"""Filter injections

Copyright (C) CSIRO 2023
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import logging
import yaml
from astropy.coordinates import SkyCoord
import astropy.units as u

from craco.candpipe.candpipe import ProcessingStep
from craco.candpipe.steps import catalog_cross_match

log = logging.getLogger(__name__)

__author__ = '''Yuanming Wang <yuanmingwang@swin.edu.au>;
                Joscha Jahns-Schindler <jjahnsschindler@swin.edu.au>;
                Yu Wing Joshua Lee <ylee2156@uni.sydney.edu.au>;'''


def get_parser():
    '''
    Create and return an argparse.ArgumentParser with the arguments you need on the commandline
    Make sure add_help=False
    '''
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
 #   parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    parser.add_argument('--injection', type=str, help='Injection yml file', default=None)
    return parser

class Step(ProcessingStep):

    step_name = "injection_filter"
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


    def __call__(self, context, cands):
        '''Filter out injections and save them sepatately.

        1. read the inject params files
        2. crossmatch the injection csv and the candidate csv based on position, dm, and time
        3. save the matched injection to injection candidate csv, then drop them from the candidate csv
        4. add the original injection info to injection candidate csv
        '''
        log.debug('Got %d candidates type=%s, columns=%s', len(cands), type(cands), cands.columns)

        p = self.pipeline

        # Only run if an injection file is given.
        if p.args.injection is None:
            log.debug('No input injection file for %s', p.cand_fname)
            return cands

        # Read yaml.
        with open(p.args.injection, 'r') as yaml_file:
            inject_params = yaml.safe_load(yaml_file)

        p.injpar = inject_params

        # Convert yaml to pandas DataFrame.
        injpar = self.convert_yaml()

        # Save converted injected file.
        if p.args.injection is not None:
            fout = p.cand_fname+'.inject.orig.csv'
            fout = os.path.join(p.args.outdir, fout)
            log.debug('Saving injected csv format file to %s', fout)
            injpar.to_csv(fout, index=False, float_format='%.8g')

        # Do the cross-matching.
        injidx, outdidx, cross_sep = self.cross_match_dm_time_coords(injpar, cands)

        # Save the found injection in the outd and original values to a new data frame.
        injs = pd.concat([cands.iloc[outdidx].reset_index(drop=True), injpar.loc[injidx].reset_index(drop=True)], axis=1)
        injs['INJ_sep'] = cross_sep
        injs = injs.sort_values('total_sample_inj')

        # Drop the saved injections from the outd.
        cands = cands.drop(cands.index[outdidx])

        if p.args.injection is not None:
            fout = p.cand_fname+'.inject.cand.csv'
            fout = os.path.join(p.args.outdir, fout)
            log.debug('Saving injected csv format file to %s', fout)
            injs.to_csv(fout, index=False, float_format='%.8g')

        return cands


    def convert_yaml(self):
        '''Convert input yaml into a pandas DataFrame, with column names
        dm_pccm3_inj, shape_inj, SNR_inj, spectrum_inj, subsample_phase_inj, tau0_inj,
        width_samps_inj, total_sample_inj, lpix_inj, mpix_inj, ra_deg_inj, dec_deg_inj.

        Additional column names added later will have the name col+'_inj'.
        '''
        injpar = self.pipeline.injpar

        # Start data frame with all "furby_props".
        injpar_df = pd.DataFrame(injpar['furby_props'])
        new_columns = {col:col+'_inj' for col in injpar_df.columns}
        injpar_df = injpar_df.rename(columns=new_columns)
        injpar_df = injpar_df.rename(columns={'dm_inj':'dm_pccm3_inj', 'snr_inj':'SNR_inj'})
        injpar_df = injpar_df.drop(columns='noise_per_sample_inj')

        injpar_df['INJ_name'] = ['INJ_' + str(n) for n in range(len(injpar_df))]
        injpar_df['total_sample_inj'] = injpar['injection_tsamps']

        # Get coordinates.
        lpixlist = np.array(injpar['injection_pixels'])[:, 0]
        mpixlist = np.array(injpar['injection_pixels'])[:, 1]

        coords = self.pipeline.get_current_pixel_to_world(lpixlist, mpixlist)
        injpar_df['lpix_inj'] = lpixlist
        injpar_df['mpix_inj'] = mpixlist
        injpar_df['ra_deg_inj'] = coords.ra.deg
        injpar_df['dec_deg_inj'] = coords.dec.deg

        return injpar_df

    def cross_match_dm_time_coords(self, injs, cands):
        '''
        Output: matched indices in the original injection, matched indices in the candidate list,
        and the separation between the injection and matched candidates
        '''
        inject_tol = self.pipeline.config['inject_tol']

        # Get dm as array.
        inj_dm = np.array(injs['dm_pccm3_inj'])[:, np.newaxis]  # Broadcasted numpy array of the dm in the injection list
        cands_dm = np.array(cands['dm_pccm3'])[np.newaxis,:]  # Broadcasted numpy array of the dm in the candidate list
        dm_threshold = inject_tol['dm_pccm3']/inject_tol['dm_frac']

        # If injection dm > dm_threshold, use threshold percentage, else use flat limit
        dm_diff = np.where(inj_dm > dm_threshold, (np.abs(inj_dm - cands_dm)/inj_dm) < inject_tol['dm_frac'],
                           np.abs(inj_dm - cands_dm) < inject_tol['dm_pccm3'])

        # Check time difference
        time_diff = np.abs((np.array(injs['total_sample_inj'])[:, np.newaxis] - np.array(cands['total_sample'])[np.newaxis, :])) < inject_tol['tsmaps']

        # Check spatial difference
        coords1 = SkyCoord(ra=np.array(injs['ra_deg_inj']), dec=np.array(injs['dec_deg_inj']), unit='deg')[:, np.newaxis]
        coords2 = SkyCoord(ra=np.array(cands['ra_deg']), dec=np.array(cands['dec_deg']), unit='deg')[np.newaxis, :]
        sep = coords1.separation(coords2).arcsec

        # Find what indices have True in all matrices
        idx_array = np.where((dm_diff & time_diff & (sep < inject_tol['srcsep'])))

        return idx_array[0], idx_array[1], sep[idx_array[0], idx_array[1]]

    def close(self):
        pass
