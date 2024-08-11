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

    step_name = "time_space_filter"
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
            outd = outd[outd['snr'] > self.pipeline.args.cluster_min_sn]
        
        return outd


    def cal_metrics(self, uncluster):

        config = self.pipeline.config 

        # for max(spatial_id) <= 0, don't need to do anything (only one spatial cluster so m3=1, m6m7=0)
        grouped = uncluster.groupby('cluster_id')
        idxs = np.array( grouped.max()['spatial_id'] > 0 ) 

        log.debug("calculate metrics for total of %s clusters", sum(idxs))

        # select spatial_id == -1 candidates 
        candidates = uncluster.loc[grouped.idxmax()['snr']][~idxs]

        # put new metrics for those clusters
        candidates['mSNR'] = 1
        candidates['mSlope'] = 0
        candidates['mSlopeSmooth'] = 0

        # put new statistics columns
        # candidates['lpix_rms'] = np.array(grouped.std()['lpix'][~idxs])
        # candidates['mpix_rms'] = np.array(grouped.std()['mpix'][~idxs])
        candidates['num_samps'] = np.array(grouped.count()['lpix'][~idxs])
        # candidates['centl'] = np.array(grouped.mean()['lpix'][~idxs])
        # candidates['centm'] = np.array(grouped.mean()['mpix'][~idxs])
        candidates['num_spatial'] = 1

        candidates = candidates.reset_index(drop=True)
        sidelobes = pd.DataFrame()

        for cluster_id in grouped.max()[idxs].index:
            # log.debug("cal metrics for cluster %s", cluster_id)
            data = uncluster[uncluster['cluster_id'] == cluster_id]

            mSNR = self.cal_mSNR(data)
            mSlope = self.cal_mSlope(data)
            mSlopeSmooth = self.cal_mSlopeSmooth(data, num=config['threshold']['num_mSlopeSmooth'])

            subgrouped = data.groupby('spatial_id')

            # select the brightest one or two spatial ids and save them into candidates
            # the rest of them are saved into sidelobes 
            if (mSlope > config['threshold']['mSlope']) or (mSlope == 0):
                # if m6 > 1, i.e. the difference between 3rd spatial cluster and 2nd spatial cluster
                # are larger than the difference between 2nd spatial cluster and 1st spatial cluster
                # or only two spatial clusters 
                # then it possibly has two sources in one time/dm cluster  
                spatial_idxs = list(subgrouped.max()['snr'].nlargest(2).index)
            else:
                spatial_idxs = list(subgrouped.max()['snr'].nlargest(1).index)

            spatial_idx_all = list(data['spatial_id'].unique())
            spatial_idx_sidelobes = [ idx for idx in spatial_idx_all if idx not in spatial_idxs ]
            
            # only select the row with maxmium SNR for each spatial id 
            uniqs_per_cluster = data.loc[ subgrouped['snr'].idxmax() ]

            uniqs_per_cluster['num_samps'] = len(data) # number of samples in total time/dm cluster (per cluster id)
            uniqs_per_cluster['num_spatial'] = len(spatial_idx_all)
            uniqs_per_cluster['mSNR'] = mSNR
            uniqs_per_cluster['mSlope'] = mSlope
            uniqs_per_cluster['mSlopeSmooth'] = mSlopeSmooth

            sidelobes_per_cluster = uniqs_per_cluster[ uniqs_per_cluster['spatial_id'].isin(spatial_idx_sidelobes) ]
            candidates_per_cluster = uniqs_per_cluster[ uniqs_per_cluster['spatial_id'].isin(spatial_idxs) ]

            candidates = pd.concat([candidates, candidates_per_cluster], ignore_index=True)
            sidelobes = pd.concat([sidelobes, sidelobes_per_cluster ], ignore_index=True)

            log.debug('for cluster id %s: num spatials %s, num sidelobes %s, mSNR %s, mSlope %s, mSlopeSmooth %s, total candidates DataFrame length %s, total sidelobes DataFrame length %s', 
                        cluster_id, len(spatial_idx_all), len(spatial_idx_sidelobes), round(mSNR, 3), round(mSlope, 3), round(mSlopeSmooth, 3), len(candidates), len(sidelobes))

        # save sidelobes 
        if self.pipeline.args.save_rfi:
            # cand_fname = os.path.join(self.pipeline.args.outdir, self.pipeline.cand_fname + '.sidelobes.csv')
            outname = os.path.join(self.pipeline.args.outdir, f"candidates.b{self.pipeline.beamno:02d}.sidelobes.csv")
            log.info('Saving sidelobes to file %s', outname)
            sidelobes.to_csv(outname)

        return candidates 



    def cal_mSlope(self, data):
        '''
        Original m6
        ratio of (3rd spatial cluster - 2nd spatial cluster) and (2nd spatial cluster - 1st spatial cluster)
        m6 > 1: first two spatial cluster are bright, and from the 3rd spatial cluster are faint, assuming first two clusters are real
        otherwise, we assume only the first cluster is real 
        '''

        snrmax = data.groupby('spatial_id').max()
        snrmax = snrmax['snr'].unique()
        snrmax = np.sort(snrmax)[::-1]

        diff = snrmax[:-1] - snrmax[1:]

        if diff.shape[0] > 1:
            return diff[1] / diff[0]

        else:
            return 0


    def cal_mSlopeSmooth(self, data, num=10):
        '''
        Original m7
        m7 -> 1: bad/possibly RFI 
        m7 -> 0: good/True candidates
        '''

        count = data.groupby('spatial_id').count()['snr']
        snrmax = data.groupby('spatial_id').max()
        snrmax['count'] = count
        snrmax = snrmax.sort_values('snr', ascending=False)
        
        # take the first <num> to avoid biased by low-SNR spaital clusters tails 
        snrmax = np.array(snrmax[['snr', 'count']].iloc[:num])
        diff = snrmax[:-1, 0] - snrmax[1:, 0]
        count = snrmax[:-1, 1]

        # only one or two spatial clusters
        if count.shape[0] <= 1:
            return 0

        # take the sum as one object can be split into two in lpix/mpix space (itself + alias)
        # when the object is in the corner/edge 
        sum = count[0] + count[1]
        # now we calcualte weighted difference by count 
        mainlobes = (diff[0] * count[0] + diff[1] * count[1]) / sum

        # RFI (first three spatial clusters have same max SNR)
        if mainlobes == 0:
            return -1
        
        # > 3 spatial clusters 
        elif diff.shape[0] > 2:
            # now we calculate sidelobes as the mean of difference of spatial clusters tails
            sidelobes = np.mean(diff[2:], 0)        
            # now the metric is ratio of the mean difference at sidelobes (tails) 
            # and weighted mean difference at brightest clusters 
            return sidelobes / mainlobes

        # have 3 spatial clusters 
        else:
            # cannot caluclate sidelobes, assume it's good 
            return 0

    
    def cal_mSNR(self, data):
        '''
        original m3 -> the concentration of SNR
        m3 -> 1: good/real candidates
        m3 -> 0: bad/possibly RFI 
        '''
        grouped_SNR = data.groupby('spatial_id').sum()['snr']

        return grouped_SNR.max() / grouped_SNR.sum()


    def classify(self, candidates):

        config = self.pipeline.config 
        cand_fname = os.path.join(self.pipeline.args.outdir, self.pipeline.cand_fname)

        # potential candidates
        cand_ind = (candidates['mSlopeSmooth'] >= 0) & (candidates['mSlopeSmooth'] < config['threshold']['mSlopeSmooth'])
        # definative RFIs
        rfi_ind = (candidates['mSNR'] < config['threshold']['mSNR'])
        # noise 
        noise_ind = self.noise(candidates)
        # Central ghosts
        cet_ind = self.central_ghost(candidates)
        # super bright sources 
        brg_ind = (candidates['snr'] >= config['threshold']['max_snr'])

        # final candidates
        cand_ind_fin = (cand_ind & (~rfi_ind) & (~cet_ind) & (~noise_ind) ) | brg_ind

        if self.pipeline.args.save_rfi:
            # for all non-candidates, put them into rfi dataframe and do further classification 
            candidates_rfi = candidates[~cand_ind_fin]
            candidates_rfi = self.classify_rfi(candidates_rfi)
            # fname = os.path.join(self.pipeline.args.outdir, self.pipeline.cand_fname + '.rfi.csv')
            outname = os.path.join(self.pipeline.args.outdir, f"candidates.b{self.pipeline.beamno:02d}.rfi.csv")
            log.info('Saving selected rfi to file %s', outname)
            candidates_rfi.to_csv(outname)

        # final candidates
        candidates_fin = candidates[cand_ind_fin]

        return candidates_fin 


    def classify_rfi(self, cands):
        '''
        1. central ghost
        2. if not central ghost, check if it's noise 
        3. rest of them are RFI 
        '''
        conditions = [
            self.central_ghost(cands), 
            self.noise(cands), 
        ]

        classes = [
            'CGHOST', 
            'NOISE'
        ]

        cands.loc[:, 'LABEL'] = np.select(conditions, classes, default='RFI')
        
        return cands


    def central_ghost(self, candidates):
        condition = (candidates['lpix'] >= 126) & (candidates['lpix'] <= 130) & (candidates['mpix'] >= 126) & (candidates['mpix'] <= 130)
        return condition.values

    def noise(self, candidates):
        config = self.pipeline.config 
        condition = (candidates['num_samps'] <= config['threshold']['num_samps'])
        return condition.values

    def format_dataframe(self, data):
        '''
        make certain dtype, e.g. SNR to :.1f lpix to int 
        '''
        dformat = self.pipeline.config['dformat']
        colint = dformat['colint']
        colfloat = dformat['colfloat']

        for ind in colint:
            try:
                data[ind] = data[ind].astype(int)
            except:
                None
        
        for ind in colfloat:
            try:
                data = data.round(ind)
            except:
                None
        
        # # make some columns as int
        # data[colint] = data[colint].astype(int)

        # # make some columns as float
        # data = data.round(colfloat)

        return data


    def close(self):
        pass
