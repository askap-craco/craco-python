#!/usr/bin/env python
"""
Clusters the input

Copyright (C) CSIRO 2023
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.candpipe.candpipe import ProcessingStep

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au> and others put your name here"


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
        #    outd = outd[outd['snr'] > self.pipeline.args.cluster_min_sn]
        

        return outd

    def close(self):
        pass
