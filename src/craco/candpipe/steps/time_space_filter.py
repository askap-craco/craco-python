#!/usr/bin/env python
"""
Clusters the input

1. Get the rawcat (with new column cluster_id and spatial_id) from last step 
2. calcualte m3, m7 for each cluster_id
3. select candidates using m3 and m7 
4. for each cluster_id, if m6 (>1), reserve the first two brightest spatial clusters, 
otherwise just pick the brightest cluster 
5. put the rest of candidates into rfi catalogue 
6. put central ghost into cet ghost catalogue 
7. save rfi.csv and cet.csv 

Copyright (C) CSIRO 2023
"""

import pandas as pd
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.candpipe.candpipe import ProcessingStep

log = logging.getLogger(__name__)

__author__ = """Keith Bannister <keith.bannister@csiro.au>; 
                Yuanming Wang <yuanmingwang@swin.edu.au>"""


def get_parser():
    '''
    Create and return an argparse.ArgumentParser with the arguments you need on the commandline
    Make sure add_help=False
    '''
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    return parser


class Step(ProcessingStep):

    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        p = self.pipeline
        # You might want to use some of these attributes

    #    log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
    #              p.srcdir, p.beamno, p.cand_fname, p.uvfits_fname,
    #              p.cas_fname, p.ics_fname, p.pcb_fname, p.args)
                  

    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate dataframe and returns a new dataframe of classified data
        Hopefully with fewer, better quality candidtes
        '''

        
        log.debug('Got %d candidates type=%s, columns=%s', len(ind), type(ind), ind.columns)
        
        # calculate m3 and m7 and other metrics 
        # do further clustering split (e.g. select first two spatial clusters for any m6>1)
        candidates = self.cal_metrics(ind)

        # filtering RFI based on metrics 
        candidates = self.classify(candidates)

        try:
            # format dataframe
            outd = self.format_dataframe(candidates)
            log.info('Formating dataframe...')
        except:
            outd = candidates
            log.warning('Cannot format dataframe...')
        
        # apply command line argument for minimum S/N and only return those values
        if self.pipeline.args.cluster_min_sn is not None:
            outd = outd[outd['SNR'] > self.pipeline.args.cluster_min_sn]
        
        return outd


    def cal_metrics(self, uncluster):

        config = self.pipeline.config 

        # for max(spatial_id) <= 0, don't need to do anything (only one spatial cluster so m3=1, m6m7=0)
        grouped = uncluster.groupby('cluster_id')
        idxs = np.array( grouped.max()['spatial_id'] > 0 ) 

        log.debug("calculate metrics for total of %s clusters", sum(idxs))

        # select spatial_id == -1 candidates 
        candidates = uncluster.loc[grouped.idxmax()['SNR']][~idxs]

        # put new metrics for those clusters
        candidates['mSNR'] = 1
        candidates['mSlope'] = 0
        candidates['mSlopeSmooth'] = 0

        # put new statistics columns
        candidates['lpix_rms'] = np.array(grouped.std()['lpix'][~idxs])
        candidates['mpix_rms'] = np.array(grouped.std()['mpix'][~idxs])
        candidates['num_samps'] = np.array(grouped.count()['lpix'][~idxs])
        candidates['centl'] = np.array(grouped.mean()['lpix'][~idxs])
        candidates['centm'] = np.array(grouped.mean()['mpix'][~idxs])
        candidates['num_spatial'] = 1

        candidates = candidates.reset_index(drop=True)

        j = len(candidates)

        for cluster_id in grouped.max()[idxs].index:
            # log.debug("cal metrics for cluster %s", cluster_id)
            data = uncluster[uncluster['cluster_id'] == cluster_id]

            mSNR = self.cal_mSNR(data)
            mSlope = self.cal_mSlope(data)
            mSlopeSmooth = self.cal_mSlopeSmooth(data, num=config['threshold']['num_mSlopeSmooth'])

            subgrouped = data.groupby('spatial_id')

            if (mSlope > config['threshold']['mSlope']) or (mSlope == 0):
                spatial_idxs = list(subgrouped.max()['SNR'].nlargest(2).index)
            else:
                spatial_idxs = list(subgrouped.max()['SNR'].nlargest(1).index)

            for spatial_id in spatial_idxs:
                max_ind = subgrouped.idxmax()['SNR'].loc[spatial_id]
                candidates = pd.concat([candidates, data[ data.index == max_ind]], axis=0, ignore_index=True)

                candidates.loc[j, 'lpix_rms'] = subgrouped.std()['lpix'].loc[spatial_id]
                candidates.loc[j, 'mpix_rms'] = subgrouped.std()['mpix'].loc[spatial_id]
                candidates.loc[j, 'num_samps'] = subgrouped.count()['lpix'].loc[spatial_id]
                candidates.loc[j, 'centl'] = subgrouped.mean()['lpix'].loc[spatial_id]
                candidates.loc[j, 'centm'] = subgrouped.mean()['mpix'].loc[spatial_id]
                candidates.loc[j, 'num_spatial'] = data['spatial_id'].max() + 1
                candidates.loc[j, 'mSNR'] = mSNR
                candidates.loc[j, 'mSlope'] = mSlope
                candidates.loc[j, 'mSlopeSmooth'] = mSlopeSmooth

                j += 1

        return candidates 



    def cal_mSlope(self, data):
        '''
        Original m6
        '''

        snrmax = data.groupby('spatial_id').max()
        snrmax = snrmax['SNR'].unique()
        snrmax = np.sort(snrmax)[::-1]

        diff = snrmax[:-1] - snrmax[1:]

        if diff.shape[0] > 1:
            return diff[1] / diff[0]

        else:
            return 0


    def cal_mSlopeSmooth(self, data, num=10):
        '''
        Original m7
        '''

        count = data.groupby('spatial_id').count()['SNR']
        snrmax = data.groupby('spatial_id').max()
        snrmax['count'] = count
        snrmax = snrmax.sort_values('SNR', ascending=False)
        
        snrmax = np.array(snrmax[['SNR', 'count']].iloc[:num])
        diff = snrmax[:-1, 0] - snrmax[1:, 0]
        count = snrmax[:-1, 1]

        # only one or two spatial clusters
        if count.shape[0] <= 1:
            return 0

        sum = count[0] + count[1]
        mainlobes = (diff[0] * count[0] + diff[1] * count[1]) / sum

        # RFI (first three spatial clusters have same max SNR)
        if mainlobes == 0:
            return -1
        
        # > 3 spatial clusters 
        elif diff.shape[0] > 2:
            sidelobes = np.mean(diff[2:], 0)        
            return sidelobes / mainlobes

        # have 3 spatial clusters 
        else:
            # return diff[-1]/diff[0]
            return 0

    
    def cal_mSNR(self, data):
        '''
        original m3
        '''
        grouped_SNR = data.groupby('spatial_id').sum()['SNR']

        return grouped_SNR.max() / grouped_SNR.sum()


    def classify(self, candidates):

        config = self.pipeline.config 
        cand_fname = os.path.join(self.pipeline.args.outdir, self.pipeline.cand_fname)

        # potential candidates
        cand_ind = (candidates['mSlopeSmooth'] >= 0) & (candidates['mSlopeSmooth'] < config['threshold']['mSlopeSmooth'])
        # potential RFIs
        rfi_ind = (candidates['mSNR'] < config['threshold']['mSNR']) | (candidates['num_samps'] <= config['threshold']['num_samps'])
        # potential Central ghosts
        cet_ind = (candidates['lpix'] >= 126) & (candidates['lpix'] <= 130) & (candidates['mpix'] >= 126) & (candidates['mpix'] <= 130)
        # potential super bright sources 
        brg_ind = (candidates['SNR'] >= config['threshold']['max_snr'])

        # final candidates
        cand_ind_fin = (cand_ind & (~rfi_ind) & (~cet_ind)) | brg_ind

        if self.pipeline.args.save_rfi:
            candidates_rfi = candidates[~cand_ind_fin]
            log.info('Saving selected rfi to file %s.rfi.csv', cand_fname)
            candidates_rfi.to_csv(cand_fname + ".rfi.csv")

        # others | which is suppose to be candidates! 
        candidates_fin = candidates[cand_ind_fin]

        return candidates_fin 


    def format_dataframe(self, data):
        '''
        make certain dtype, e.g. SNR to :.1f lpix to int 
        '''
        dformat = self.pipeline.config['dformat']
        colint = dformat['colint']
        colfloat = dformat['colfloat']
        
        # make some columns as int
        data[colint] = data[colint].astype(int)

        # make some columns as float
        data = data.round(colfloat)

        return data


    def close(self):
        pass
