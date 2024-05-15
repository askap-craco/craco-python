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
import traceback
from astropy.io import fits
from astropy.wcs import WCS
from craco.candidate_writer import CandidateWriter
import pandas as pd

from . import steps
from craco.plot_cand import load_cands

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def beam_from_cand_file(candfile):
    #beamno = int(candfile[candfile.find('b')+1 : candfile.find('b') + 3])
    '''
    Get beam from a file that looks like this
    candfile = '/data/craco/craco/wan348/benchmarking/SB61584.n500.candidates.b24.txt'
    '''
    beamno = int(candfile.split('.')[-2][1:])
    return beamno

def load_default_config():
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

def copy_best_cand(cand_pd, cand_np):
    '''
    Copy data frame candidates to numpy candidates
    There are more that can fit, then we take the top N sorted by snr
    If there are less then re remaining will have snr=-1
    Ignore missing columns
    '''
    #assert len(cand_pd) == len(cand_np)
    # Copy candidates from pandas data frame to numpy array
                
    MAXN = len(cand_np)
    N = len(cand_pd)
    top_cand = cand_pd.sort_values(by='snr', ascending=False).head(MAXN)
    cand_np[N:]['snr'] = -1
    
    for field in cand_np.dtype.fields.keys():
        try:
            cand_np[field][:N] = top_cand[field].iloc[:N]
        except KeyError:
            pass

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
    def __init__(self, beamno, args, config, src_dir='.', anti_alias=False):
        '''
        
        
        :param  beamno: beam number of reading/writing files. 
        If beamno is an integer, it uses that. src_dir must be specified
        if otherwise, it derives the beamno and src_dir assuming beamno is the candidates file path

        :param args: Command line arguments - if None loads default arguments
        :param config: YAML config - if None, uses load_default_config()
        :param src_dir: root directory for input files. If None assumes '.' unles src dir is from candfile
        :param anti_alias: if anti_alias is True it will try anti aliasing. If the PSF doesn't exists it will error.
        If anti_alias is False, it won't anti alias
        if anti_alias is None, it will anti_alias if the PSF exists
        '''
        if args is None:
            args = get_parser().parse_args([])
        self.args = args

        if config is None:
            config = load_default_config()

        if isinstance(beamno, int):
            self.beamno = beamno
            if src_dir is None:
                src_dir = '.'
            self.cand_fname = 'NONE'
        else:
            self.cand_fname = beamno
            assert os.path.exists(self.cand_fname), f'{self.cand_fname} does not exist'
            self.beamno = beam_from_cand_file(self.cand_fname)
            if src_dir is None:
                src_dir = os.path.dirname(self.cand_fname)

        # self.beamno = int(candfile.replace('candidates.txtb',''))
        assert 0<= self.beamno < 36
        assert args is not None
        assert config is not None
        self.config = config
        self.output_npy_dtype = None

        self.srcdir = src_dir
        assert os.path.isdir(src_dir), f'{src_dir} is not a directory'
        self.uvfits_fname = self.get_file( f'b{self.beamno:02d}.uvfits')
        self.cas_fname = self.get_file( f'cas_b{self.beamno:02d}.fil')
        self.ics_fname = self.get_file( f'cas_b{self.beamno:02d}.fil')
        self.pcb_fname = self.get_file( f'pcb{self.beamno:02d}.fil')
        self.psf_fname = self.get_file( f'psf.beam{self.beamno:02d}.iblk0.fits')
        psf_exists = os.path.isfile(self.psf_fname)
        if anti_alias is None:
            anti_alias = psf_exists

        self.anti_alias = anti_alias

        if anti_alias:
            if psf_exists:
                self.load_psf_from_file(0)
            else:
                pass
                # hope that someone calls set_current_psf()

            self.steps = [
                steps.cluster.Step(self),
                steps.time_space_filter.Step(self), 
                steps.catalog_cross_match.Step(self),
                steps.alias_filter.Step(self), 
                steps.injection_filter.Step(self), 
            ]
        else:
            self.steps = [
                steps.cluster.Step(self),
                steps.time_space_filter.Step(self), 
                steps.catalog_cross_match.Step(self),
            ]

        log.debug('srcdir=%s beamno=%s candfile=%s uvfits=%s cas=%s ics=%s pcb=%s arguments=%s',
                  self.srcdir, self.beamno, self.cand_fname, self.uvfits_fname,
                  self.cas_fname, self.ics_fname, self.pcb_fname, self.args)

    @staticmethod
    def from_candfile(candfile, args, config):
        beamno = beam_from_cand_file(candfile)
        pipe = Pipeline(beamno, args, config, os.path.dirname(candfile))
        pipe.cand_fname= candfile
        return pipe
            
        
    def get_file(self, fname):
        '''
        Returns full path relative to candidate file for a file with the given filename
        Prints warning if file doesn't exist
        '''
        full_path = os.path.join(self.srcdir, fname)
        if not os.path.exists(full_path):
            log.warning('Expected input file doesnt exist %s', full_path)

        return full_path

    def set_current_psf(self, iblk, hdr):        
        self.psf_header = hdr
        self.curr_wcs = WCS(hdr)
        self.curr_psf_iblk = iblk
        
        return hdr

    def load_psf_from_file(self, iblk):
        psf_fname = self.get_file( f'psf.beam{self.beamno:02d}.iblk{iblk:d}.fits')
        hdr = fits.getheader(psf_fname)
        self.curr_psf_file = psf_fname
        self.set_current_psf(iblk, hdr)


    def create_dir(self):
        outdir = self.args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        else:
            log.debug('Directory %s exists.', outdir)

    
    def run(self):
        cand_in = load_cands(self.cand_fname, fmt='pandas')
        log.debug('Loaded %d candidates from %s beam=%d. Columns=%s', len(cand_in), self.cand_fname, self.beamno, cand_in.columns)

        # create a directory to store output files 
        self.create_dir()
        cand_out = self.process_block(cand_in)
        log.info('Produced %s candidates in total', len(cand_out))
        fout = os.path.join(self.args.outdir, self.cand_fname+'.uniq.csv')
        log.info('Saving final candidates to %s', fout)
        cand_out.to_csv(fout)
        log.debug('Saved final candidates to %s', fout)

        self.close()

        return cand_out

    def get_current_fov(self):
        '''
        Return FoV in degrees as a tuple (ra_fov, dec_fov)
        
        TODO: Should proably use WCS pixel_to_world for the extremes, but Yuanming said that worked poorly.
        '''
        h = self.psf_header
        ra_fov = np.abs(h['NAXIS1'] * h['CDELT1'])
        dec_fov = np.abs(h['NAXIS2'] * h['CDELT2'])

        return (ra_fov, dec_fov)
    
    def close(self):
        for step in self.steps:
            step.close()
    
    def convert_np_to_df(self, npin):
        '''
        Convert  numpy array to data frame.
        Vivek says change this to include additional columns (that will be inevitably created by the candpipe during processing steps) to ipmrove speed. FOr now it just naievely converts
        '''
        assert isinstance(npin, np.ndarray)
        assert npin.dtype in (CandidateWriter.out_dtype , CandidateWriter.out_dtype_short)
        df = pd.DataFrame(npin)
        return df
    
    def convert_df_to_np(self, dfin):
        '''
        Converts pandas dataframe to a numpy recordarray
        '''
        if self.output_npy_dtype is None:
            dtype_list = []
            for item in dfin.dtypes.items():
                dtype_list.append(item)

            self.output_npy_dtype = np.dtype(dtype_list)
        
        cands_npy_array = dfin.to_numpy()
        cands_npy_array = np.array(np.rec.fromarrays(cands_npy_array.transpose(), names = self.output_npy_dtype.names).astype(dtype=self.output_npy_dtype).tolist(), dtype=self.output_npy_dtype)
        return cands_npy_array

    def process_block(self, cand_in, cand_out_buf=None):
        '''
        Candidates: is assumed to be a pandas data frame or numpy array of type out_dtype
        cand_out_buf - if it's a np.dtype=cand_out it will put the best N candidates into that the dataframe to that cand out.
        '''


        if isinstance(cand_in, np.ndarray):
            cand_in = self.convert_np_to_df(cand_in)

        if len(cand_in) == 0:
            cand_out = pd.DataFrame(np.zeros(0, dtype=CandidateWriter.out_dtype))
        else:
            iblk = cand_in['iblk']
            iblk0 = iblk[0]
            # Sometimes for testing we send through a giant batch.
            # We'll only load the PSF frm the first candidate, if possible.
            # assert np.all(cand_in['iblk'] == iblk0), f'Should only get 1 iblk at a time {iblk} != {cand_in["iblk"]}'
            try:
                self.load_psf_from_file(iblk0)
                log.info('Loaded new PSF for iblk=%d', iblk0)
            except FileNotFoundError: # No PSF available. Oh well. maybe next year.
                pass

            for istep, step in enumerate(self.steps):
                cand_out = step(self, cand_in)
                stepname = step.__module__.split('.')[-1]
                log.debug('Step "%s" produced %d candidates maxsnr=%0.2f', stepname, len(cand_out), cand_out['snr'].max())
                if self.args.save_intermediate:
                    fout = self.cand_fname+f'.{stepname}.i{istep}.csv'
                    fout = os.path.join(self.args.outdir, fout)
                    log.debug('Saving step %s i=%d to %s', stepname, istep, fout)
                    cand_out.to_csv(fout)

                cand_in = cand_out

        if cand_out_buf is not None:
            copy_best_cand(cand_out, cand_out_buf)
        
        return cand_out


def get_parser():
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
    parser.add_argument(dest='files', nargs='*')
    parser.set_defaults(verbose=False)
    return parser

def run_with_args(args, config):
    for f in args.files:
        try:
            p = Pipeline(f, args, config, anti_alias=None)
            p.run()
        except:
            log.error(traceback.format_exc())
            log.error(f"failed to run candpipe on {f}... aborted...")

def _main():
    parser = get_parser()
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

    log.debug("Executing candpipe...")

    config = load_default_config()

    run_with_args(args, config)

    

if __name__ == '__main__':
    _main()
