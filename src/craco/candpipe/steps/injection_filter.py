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

from craco.candpipe.candpipe import ProcessingStep
from craco.candpipe.steps import catalog_cross_match

log = logging.getLogger(__name__)

__author__ = '''Yuanming Wang <yuanmingwang@swin.edu.au>;
                Joscha Jahns-Schindler <jjahnsschindler@swin.edu.au>; '''


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
        #from IPython import embed
        #embed()

        p = self.pipeline

        if self.pipeline.args.injection is None: 
            log.warning('No input injection file for %s', p.cand_fname)
            return ind

        ## 1. read the inject params files
        ## 2. insert new columns, INJ_name, ING_sep (name is just inj_1, inj_2 etc.)
        ## 3. do crossmatch use coordinates, dm, and time (set a bit tolerance)
        ## 4. do crossmatch for each injected signal, 
        ## 5. if there is a detection, add inj info into this row 
        ## 6. if there is no detection, add inj info to a new row 


        # read yaml
        with open(self.pipeline.args.injection, 'r') as yaml_file:
            inject_params = yaml.safe_load(yaml_file)

        p.injpar = inject_params

        # convert yaml to pandas DataFrame 
        injpar, coords = self.convert_yaml()

        # save converted injected file
        if p.args.injection is not None:
            fout = p.cand_fname+'.inject.orig.csv'
            fout = os.path.join(self.pipeline.args.outdir, fout)
            log.debug('Saving injected csv format file to %s', fout)
            injpar.to_csv(fout, index=False, float_format='%.8g')

        outd = ind
        injdic = pd.DataFrame()

        for idx in range(len(injpar)):
            injcat = injpar.iloc[[idx]]
            injcoord = coords[[idx]]
            print(injcoord)

            # crossmatch (spatially)
            outd = catalog_cross_match.Step(p).cross_matching(candidates=outd, 
                                        catalogue=injcat, 
                                        coord=injcoord, 
                                        threshold=p.config['inject_tol']['srcsep'], 
                                        col_prefix='INJ', 
                                        key='name')

            # crossmatch (time/dm)
            outd, injdic = self.check_time_dm(outd, injcat, injdic)

        if p.args.injection is not None:
            fout = p.cand_fname+'.inject.cand.csv'
            fout = os.path.join(self.pipeline.args.outdir, fout)
            log.debug('Saving injected csv format file to %s', fout)
            injdic.to_csv(fout, index=False, float_format='%.8g')
               
        return outd


    def convert_yaml(self):
        # convert input yaml into a pandas DataFrame, with column names
        # name, lpix, mpix, ra_deg, dec_deg, tsamps, dm, snr

        injpar = self.pipeline.injpar
        injpar_dic = pd.DataFrame()

        lpixlist = np.array(injpar['injection_pixels'])[:, 0]
        mpixlist = np.array(injpar['injection_pixels'])[:, 1]

        coords = self.pipeline.wcs_info.pixel_to_world(lpixlist, mpixlist)
        
        injpar_dic['name'] = ['INJ_' + str(n) for n in range(len(lpixlist))]
        injpar_dic['lpix'] = lpixlist
        injpar_dic['mpix'] = mpixlist
        injpar_dic['ra_deg'] = coords.ra.deg
        injpar_dic['dec_deg'] = coords.dec.deg
        injpar_dic['total_sample'] = injpar['injection_tsamps']
        injpar_dic['dm_pccm3'] = [injpar['furby_props'][n]['dm'] for n in range(len(lpixlist))]
        injpar_dic['snr'] = [injpar['furby_props'][n]['snr'] for n in range(len(lpixlist))]

        return injpar_dic, coords


    def check_time_dm(self, ind, injpar, injdic):
        # for each crossmatched INJ signal, check if their time/dm are in tolerant range
        # if not, remove this row and save to injection file 
                
        outd = pd.merge(ind, injpar[['name', 'total_sample', 'dm_pccm3', 'snr']], left_on='INJ_name', right_on='name', 
                        sort=False, how='outer', suffixes=("", '_inj'))

        outd['INJ_name'] = outd['name']

        ind_time = (outd['total_sample_inj'] - outd['total_sample']).abs() > self.pipeline.config['inject_tol']['tsmaps']
        ind_dm = (outd['dm_pccm3_inj'] - outd['dm_pccm3']).abs() > self.pipeline.config['inject_tol']['dm_pccm3']

        idx = ind_time | ind_dm

        outd.loc[idx, 'INJ_name'] = None
        outd.loc[idx, 'INJ_sep'] = None

        # put those rows with INJ_name into another file
        injdic = pd.concat([injdic, outd[ ~outd['INJ_name'].isna() ] ], ignore_index=True)
        outd = outd[ outd['INJ_name'].isna() ]

        # remove injpar dic
        outd = outd.drop(columns=['name', 'total_sample_inj', 'dm_pccm3_inj', 'snr', 'INJ_name', 'INJ_sep'])
        injdic = injdic.drop(columns=['name'])

        return outd, injdic


    def close(self):
        pass
