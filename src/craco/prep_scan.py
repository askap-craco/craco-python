#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craft import fcm2calc, calc11
import pickle
import shutil
import subprocess
from astropy.coordinates import SkyCoord
from astropy.time import Time

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class ScanPrep:
    def __init__(self, targname:str, beam_phase_centers:SkyCoord, fcmfile:str, outdir:str, start:Time, stop:Time):
        assert len(beam_phase_centers) == 36
            
        self.outdir = outdir
        self.targname = targname
        self.beam_phase_centers = beam_phase_centers
        self.fcmfile = fcmfile
        self.start = start
        self.stop = stop

    def save(self):
        os.makedirs(self.outdir, exist_ok=True)
        shutil.copyfile(self.fcmfile, self.local_fcm_file_name)
        self.fcmfile = self.local_fcm_file_name
        self.write_calcfiles()
        self.run_calc()
        with open(self.index_file_name(self.outdir), 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(outdir):
        with open(ScanPrep.index_file_name(outdir), 'rb') as fin:
            prep = pickle.load(fin)

        return prep

    @staticmethod
    def index_file_name(outdir):
        return os.path.join(outdir, 'index.pkl')

    @property
    def local_fcm_file_name(self):
        return os.path.join(self.outdir, 'fcm.txt')

    def beamdir(self, ibeam):
        d = os.path.join(self.outdir, f'beam{ibeam:02d}')
        os.makedirs(d, exist_ok=True)
        return d
    
    def calc_file_name(self, ibeam):
        return os.path.join(self.beamdir(ibeam), 'target.calc')

    def results_file_name(self, ibeam):
        return os.path.join(self.beamdir(ibeam), 'target.im')

    def difx_log_file_name(self, ibeam):
        return os.path.join(self.beamdir(ibeam), 'difx.log')

    def results_file(self, ibeam):
        return calc11.ResultsFile(self.results_file_name(ibeam))

    def write_calcfiles(self):
        ant_data, antnos = fcm2calc.load_parset(self.fcmfile)
        start = self.start
        stop = self.stop
        targname = self.targname
        for ibeam, skycoord in enumerate(self.beam_phase_centers):
            calcfile = calc11.CalcFile.from_time_src(start, stop, skycoord, targname)
            calcfile.add_antdata(ant_data, antnos)
            calcfile.add_eops()
            calcfile.writeto(self.calc_file_name(ibeam))

    @property
    def nbeams(self):
        return len(self.beam_phase_centers)

    def run_calc(self):
        '''
        Run calc in parallel for all beams
        '''
        logfiles = [open(self.difx_log_file_name(ibeam), 'wb') for ibeam in range(self.nbeams)]
        commands = [subprocess.Popen(['difxcalc', \
                                      self.calc_file_name(ibeam)],
                                     stdout=logfile,
                                     stderr=logfile
        )
                                     for ibeam, logfile in enumerate(logfiles)]

        for ibeam, cmd in enumerate(commands):
            cmd.wait()
            if cmd.returncode != 0:
                raise ValueError(f'difxcalc failed {cmd.returncode}')
            
            results_file = self.results_file(ibeam)
            assert os.path.exists(results_file.fname), f'Calc didnt create file {results_file}'

        for l in logfiles:
            l.close()
            

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--fcm', required=True, help='FCM file')
    parser.add_argument('-a','--antenna', type=int, required=True, help='Antenna number ot use for EPICS query')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()