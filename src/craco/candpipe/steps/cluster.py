#!/usr/bin/env python
"""
Clusters the input

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
    parser.add_argument('--cluster-min-sn', type=float, help='Minimum S/N of cluster output', default=None)
    return parser


class Step(ProcessingStep):
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
        #from IPython import embed
        #embed()

        # do first clustering - time/dm/boxcar
        candidates, clustered = self.dbscan(ind)

        # do second clustering - spatial/mpix/lpix 
        candidates = self.spatial_clustering(candidates, clustered)
        
        # do classification - rule out RFI
        outd = self.classify(candidates)

        # apply command line argument for minimum S/N and only return those values
        if self.pipeline.args.cluster_min_sn is not None:
            outd = outd[outd['SNR'] > self.pipeline.args.cluster_min_sn]
        

        return outd

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
    

    def dbscan(self, data):

        ### reset the index just in case...
        data = data.reset_index(drop=True)

        # rescale data
        # => rescaled_data is numpy.ndarray; 
        rescaled_data, reference_eps_param = self.rescale_data(data, self.pipeline.config['eps'])

        cls = DBSCAN(eps=reference_eps_param, 
                     min_samples=self.pipeline.config['min_samples']).fit(rescaled_data)   
        
        # data is the original dataframe
        data["cluster_id"] = cls.labels_

        cand_group = data.groupby("cluster_id")

        unique_cand = data.loc[cand_group["SNR"].idxmax()]
        unique_cand["lpix_rms"] = cand_group["lpix"].std().to_numpy()
        unique_cand["mpix_rms"] = cand_group["mpix"].std().to_numpy()
        unique_cand["centl"] = cand_group["lpix"].mean().to_numpy()
        unique_cand["centm"] = cand_group["mpix"].mean().to_numpy()
        unique_cand["num_samps"] = cand_group.size().to_numpy()

        return unique_cand, data


    def spatial_clustering(self, candidates, clustered):
        '''
        Do second clustering, and generate spatial coherent clusters
        '''
        config = self.pipeline.config 

        # add one extra column - how many spatial clusters for each time/dm cluster

        num_spatial = []
        labels = []

        for i in range(len(candidates)):
            if candidates['lpix_rms'][i] > config['threshold']['lpix_rms'] and candidates['mpix_rms'][i] > config['threshold']['mpix_rms'] :
                data = clustered[clustered['cluster_id'] == i]
                rescaled_data, reference_eps_param = self.rescale_data(data, config['eps2'])
                cls = DBSCAN(eps=reference_eps_param, min_samples=config['min_samples']).fit(rescaled_data)  
                num_spatial.append(max(cls.labels_))
                labels.append(cls.labels_)
            else:
                num_spatial.append(-1)
                labels.append([0])

        candidates['num_spatial'] = num_spatial

        # ======

        candidates['cluster_id'] = range(len(candidates))

        # create a new file, adding more rows + spatial id
        candidates_new = candidates[(candidates['num_spatial'] <=0) | (candidates['num_spatial'] > config['threshold']['num_spatial'])]
        candidates_new = candidates_new.reset_index(drop=True)

        j = len(candidates_new)

        for ind in range(len(candidates)):
            
            if candidates['num_spatial'][ind] <=0 or candidates['num_spatial'][ind] > config['threshold']['num_spatial']:
                continue
            
            for i in range(max(labels[ind])+1):
                data = clustered[clustered['cluster_id'] == ind][labels[ind] == i]
                # find highest SNR row
                cand_ind = data['SNR'].idxmax()
                candidates_new.loc[j] = data.loc[cand_ind]
                candidates_new.loc[j, 'lpix_rms'] = np.std(data['lpix'])
                candidates_new.loc[j, 'mpix_rms'] = np.std(data['mpix'])
                candidates_new.loc[j, 'num_samps'] = len(data)
                candidates_new.loc[j, 'centl'] = np.mean(data['lpix'])
                candidates_new.loc[j, 'centm'] = np.mean(data['mpix'])
                candidates_new.loc[j, 'num_spatial'] = 0
                
                j += 1

        return candidates_new



    def classify(self, candidates_new):

        config = self.pipeline.config 

        # RFI
        # number of samples 
        rfi_ind = ((candidates_new['num_samps'] <= config['threshold']['num_samps']) | (candidates_new['num_spatial'] > config['threshold']['num_spatial'])) & (candidates_new['SNR'] < config['threshold']['max_snr'])
        candidates_rfi = candidates_new[rfi_ind]

        # # central ghost
        # cet_ind = (self.candidates_new['lpix'] >= 127) & (self.candidates_new['lpix'] <= 128) & (self.candidates_new['mpix'] >= 127) & (self.candidates_new['mpix'] <= 128)
        # self.candidates_cet = self.candidates_new[cet_ind]

        # others | which is suppose to be candidates! 
        candidates_fin = candidates_new[(~rfi_ind)]
        # self.candidates_fin = self.candidates_new[(~rfi_ind) & (~cet_ind)]

        return candidates_fin 


