#!/usr/bin/env python
"""
Clusters the input

1. Get the cluster id (time/dm cluster)
2. and spatial id (spatial cluster) to rawcat
3. and pass to next filtering layer
4. save rawcat.csv catalogue 

Copyright (C) CSIRO 2023
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import os
import sys
import logging
from craco.candpipe.candpipe import ProcessingStep

log = logging.getLogger(__name__)

__author__ = '''Keith Bannister <keith.bannister@csiro.au>; 
                Pavan Uttarkar <pavan.uttarkar@gmail.com>; 
                Yuanming Wang <yuanmingwang@swin.edu.au>
                Ziteng Wang <ztwang201605@gmail.com>'''

def get_parser():
    '''
    Create and return an argparse.ArgumentParser with the arguments you need on the commandline
    Make sure add_help=False
    '''
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    # parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    parser.add_argument('--save-rfi', action='store_true', help='Save removed rfi into a file')
    return parser


class Step(ProcessingStep):
    
    step_name = "cluster"
    def __init__(self, *args, **kwargs):
        '''
        Initialise and check any inputs after calling super
        '''
        super(Step, self).__init__(*args, **kwargs)
        # You might want to use some of these attributes
        p = self.pipeline
        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
                  p.srcdir, p.beamno, p.cand_fname, p.uvfits_fname,
                  p.cas_fname, p.ics_fname, p.pcb_fname, p.args)
    
                  
    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate dataframe and returns a new dataframe of classified data
        Hopefully with fewer, better quality candidtes
        '''

        log.debug('Got %d candidates type=%s, columns=%s', len(ind), type(ind), ind.columns)
        
        # do first clustering - time/dm/boxcar
        uncluster = self.time_dm_clustering(ind)

        # do second clustering - spatial/mpix/lpix 
        uncluster = self.spatial_clustering(uncluster)

        # save the raw uncluster file - intermediate products
        if self.pipeline.args.save_intermediate:
            cand_fname = os.path.join(self.pipeline.args.outdir, self.pipeline.cand_fname)
            log.info('Saving raw candfile with cluster id to file %s.rawcat.csv', cand_fname)
            uncluster.to_csv(cand_fname + ".rawcat.csv")
        
        return uncluster

    def close(self):
        pass

    def rescale_data(self, data, eps_params):
        '''
        It rescales the data values according to the eps_params specified by user
        It choses a reference dimension and rescales other dimensions based on the
        ratio of the eps_values of other dimensions to that reference dimension
        '''
        column_keys = list(eps_params.keys())
        reference_key = column_keys[0]
        reference_eps_param = eps_params[reference_key]

        rescaled_data_shape = (len(data), len(column_keys))
        rescaled_data = np.zeros(rescaled_data_shape)
        for ikey, key in enumerate(column_keys):
            eps_rescale_ratio = eps_params[key] / eps_params[reference_key]
            rescaled_data[:, ikey] = data[key] / eps_rescale_ratio

        return rescaled_data, reference_eps_param
    

    def time_dm_clustering(self, data):

        ### reset the index just in case...
        data = data.reset_index(drop=True)

        if self.pipeline.anti_alias:

            #>>>>>>
            # test for time-dependent cluster
            # Convert time to the band center.
            freq_bot = self.pipeline.psf_header['FCH1_HZ']
            chan_bw = self.pipeline.psf_header['CH_BW_HZ']
            n_chan = self.pipeline.psf_header['NCHAN']
            tsamp = self.pipeline.psf_header['TSAMP']

            freq_mid = freq_bot + n_chan/2 * chan_bw
            delay = 4.15 * data['dm_pccm3'] * ((1e9/freq_bot)**2 - (1e9/freq_mid)**2)  # freqs must be Hz, result is in ms.
            save_ts = data['total_sample'].copy()

            log.debug('Low frequency %sMHz, bandwidth %s MHz with %s channels, time resolution %s ms, central frequency %s MHz', 
                        np.round(freq_bot/1e6, 1), 
                        np.round(chan_bw/1e6, 0), n_chan, 
                        np.round(tsamp*1e3, 1), np.round(freq_mid/1e6, 1))

            data['total_sample'] -= delay/1000/tsamp

            #<<<<<

        # rescale data
        # => rescaled_data is numpy.ndarray; 
        rescaled_data, reference_eps_param = self.rescale_data(data, self.pipeline.config['eps'])


        if self.pipeline.anti_alias:
            data['total_sample_middle'] = data['total_sample'].copy()
            data['total_sample'] = save_ts            

        if len(rescaled_data) > 0: # DBSCAN doesn't like 0
            cls = DBSCAN(eps=reference_eps_param, 
                         min_samples=self.pipeline.config['min_samples']).fit(rescaled_data)   
        
            # data is the original dataframe
            data["cluster_id"] = cls.labels_
        else:
            data['cluster_id'] = []

        return data 


    def spatial_clustering(self, uncluster):
        '''
        Do second clustering, and generate spatial coherent clusters
        '''
        config = self.pipeline.config 

        # add one extra column - how many spatial clusters for each time/dm cluster
        grouped = uncluster.groupby('cluster_id').std()

        # select clusters that needs to do spatial clustering 
        idxs = np.array( (grouped['lpix'] > config['threshold']['lpix_rms']) | (grouped['mpix'] > config['threshold']['mpix_rms']) )
        uncluster['spatial_id'] = -1

        log.debug("%s time/dm clusters are doing spatial clustering...", sum(idxs))

        for i in grouped[idxs].index:
            data = uncluster[uncluster['cluster_id'] == i]
            rescaled_data, reference_eps_param = self.rescale_data(data, config['eps2'])
            cls = DBSCAN(eps=reference_eps_param, min_samples=config['min_samples']).fit(rescaled_data)  
            # sort labels by snr 
            # hmm need to think about a quick and elegent way! 
            # ok - maybe make it easier: just find the brightest spatial and replace this to 0 
            # and switch the original 0 spatial id to the original brightest spatial id) 
            ind = data['snr'].idxmax() # data index
            iloc = data.index.get_loc(ind) # data iloc 
            brightest_spatial_idx = cls.labels_[iloc]
            if brightest_spatial_idx != 0:
                log.info('for cluster id %s: original brightest spatial idx %s -> will replace it to 0', i, brightest_spatial_idx)
            else:
                log.debug('for cluster id %s: original brightest spatial idx %s -> will replace it to 0', i, brightest_spatial_idx)

            # switch spatial id - assign spatial id = 0 for the brightest spatial cluster 
            labels = cls.labels_
            mask_1 = labels==brightest_spatial_idx
            mask_2 = labels==0
            labels[ mask_1 ] = 0
            labels[ mask_2 ] = brightest_spatial_idx

            uncluster.loc[data.index, 'spatial_id'] = cls.labels_
            
        # replace spatial clusters for all other clusters to -1
        # uncluster.fillna({'spatial_id': -1}, inplace=True)
        uncluster['spatial_id'] = uncluster['spatial_id'].astype(int)

        return uncluster 

        

