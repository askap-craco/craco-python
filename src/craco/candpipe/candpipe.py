#!/usr/bin/env python
"""
Candidate pipeline super-script.

Runs the candidate pipeline for a given beam and computes the results

Copyright (C) CSIRO 20231
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import sys
import logging
from . import steps
from craco.plot_cand import load_cands

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class ProcessingStep:
    '''
    Each processing step should inherit this class
    I don't know what we should add to it yet, but it might come in handy
    '''

    name = 'generic'
    
    def __init__(self, pipeline):
        '''
        Pipeline steps should call this function with the pipeline  and then do their own
        initialisaation. e.g. opening and checking input files, etc
        '''
        self.pipeline = pipeline # keep a references to the parent pipeline


    def __call__(self, context, ind):
        '''
        Takes in the context and the input candidate table and returns a new table of classified data
        Hopefully with fewer, better quality candidtes
        '''
        raise NotImplementedError('You should override this method')


    def close(self):
        '''
        Called at the end of the run. you should close all your outputs here
        '''
        pass


class Pipeline:
    def __init__(self, candfile, args, config):
        self.args = args
        self.cand_fname= candfile
        # self.beamno = int(candfile.replace('candidates.txtb',''))
        self.beamno = int(candfile[candfile.find('b')+1 : candfile.find('b') + 3])
        self.srcdir = os.path.dirname(self.cand_fname)
        self.uvfits_fname = self.get_file( f'b{self.beamno:02d}.uvfits')
        self.cas_fname = self.get_file( f'cas_b{self.beamno:02d}.fil')
        self.ics_fname = self.get_file( f'cas_b{self.beamno:02d}.fil')
        self.pcb_fname = self.get_file( f'pcb{self.beamno:02d}.fil')
        self.config = config

        self.steps = [
            steps.cluster.Step(self),
            # steps.time_space_filter.Step(self),
            steps.catalog_cross_match.Step(self),
            # steps.check_filterbanks.Step(self),
            # steps.check_visibilities.Step(self),
        ]
        
        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
                  self.srcdir, self.beamno, self.cand_fname, self.uvfits_fname,
                  self.cas_fname, self.ics_fname, self.pcb_fname, self.args)
        
    def get_file(self, fname):
        '''
        Returns full path relative to candidate file for a file with the given filename
        Prints warning if file doesn't exist
        '''
        full_path = os.path.join(self.srcdir, fname)
        if not os.path.exists(full_path):
            log.warning('Expected input file doesnt exist %s', full_path)

        return full_path


    def create_dir(self):
        outdir = self.args.outdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            log.debug('Create new directory %s', outdir)
        else:
            log.debug('Directory %s exists.', outdir)

    
    def run(self):
        cand_in = load_cands(self.cand_fname, fmt='pandas')
        log.debug('Loaded %d candidates from %s beam=%d. Columns=%s', len(cand_in), self.cand_fname, self.beamno, cand_in.columns)
        self.create_dir()

        for istep, step in enumerate(self.steps):
            cand_out = step(self, cand_in)
            stepname = step.__module__.split('.')[-1]
            log.debug('Step "%s" produced %d candidates', stepname, len(cand_out))
            if self.args.save_intermediate:
                fout = self.cand_fname+f'.{stepname}.i{istep}.csv'
                fout = os.path.join(self.args.outdir, fout)
                log.debug('Saving step %s i=%d to %s', stepname, istep, fout)
                cand_out.to_csv(fout)

            cand_in = cand_out

        log.info('Produced %s candidates in total', len(cand_out))
        fout = os.path.join(self.args.outdir, self.cand_fname+'.uniq.csv')
        log.info('Saving final candidates to %s', fout)
        cand_out.to_csv(fout)
        log.debug('Saved final candidates to %s', fout)

        for step in self.steps:
            step.close()

        return cand_out
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # get parsers from every step and add it to a global parser
    parents = []
    for stepname in steps.__all__:
        stepmod = getattr(steps, stepname)
        parse_func = getattr(stepmod, 'get_parser')
        parser = None if parse_func is None else parse_func()
        if parser is not None:
            parents.append(parser)
    
    parser = ArgumentParser(description='Run the candidate pipeline', formatter_class=ArgumentDefaultsHelpFormatter, parents=parents)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-s','--save-intermediate', action='store_true', help='Save intermediate data tables')
    parser.add_argument('-o', '--outdir', type=str, default='.', help='output directory')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    
    args = parser.parse_args()
    if args.verbose:
        # logging.basicConfig(level=logging.DEBUG)
        logging.basicConfig(
            format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
            level=logging.DEBUG,
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        # logging.basicConfig(level=logging.INFO)
        logging.basicConfig(
            format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    for f in args.files:
        try:
            p = Pipeline(f, args, config)
            p.run()
        except:
            logging.info(f"failed to run candpipe on {f}... aborted...")

    
    

if __name__ == '__main__':
    _main()
