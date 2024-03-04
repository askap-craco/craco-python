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
from astropy import units as u
from craco.metadatafile import MetadataFile
from craco.calc_metafile import CalcMetafile
import datetime


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def touchfile(f, directory=None, check_exists=True):
    date = datetime.datetime.utcnow()
    if directory is None:
        fout = f
    else:
        fout = os.path.join(directory, f)

    if check_exists and os.path.exists(fout):
        raise ValueError(f'touchfile {fout} exists. Somehting bad has happend')
    
    with open(fout, 'w') as outfile:
        outfile.write(date.isoformat() + '\n')

    return fout


def make_scan_directories(sbid, scanid, target, craco_data_dir=None):
    if craco_data_dir is None:
        craco_data_dir = os.environ['CRACO_DATA']

    bigdir = craco_data_dir
    now = datetime.datetime.utcnow()
    nowstr = now.strftime('%Y%m%d%H%M%S')
    scandir = os.path.join(bigdir, f'SB{sbid:06}', 'scans', f'{scanid:02d}', nowstr)
    targetdir = os.path.join(bigdir, f'SB{sbid:06}', 'targets', target.replace(' ','_'))
    targetlink = os.path.join(targetdir, nowstr)
    os.makedirs(scandir)
    os.makedirs(targetdir, exist_ok=True)
    os.symlink(scandir, targetlink)
    return scandir

def first(x):
    return next(iter(x))

NANT = 36

class ScanPrep:
    def __init__(self, targname:str, sbid, scan_id, beam_phase_centers:SkyCoord, fcmfile:str, outdir:str, start:Time, stop:Time, ant_numbers):
        '''
        ant_numbers is a 1 based np array of antennas which should be
        included in the array
        '''
        assert len(beam_phase_centers) == 36
            
        self.outdir = outdir
        self.targname = targname
        self.beam_phase_centers = beam_phase_centers

        if fcmfile is None:
            fcmfile = os.environ['FCM']

        self.fcmfile = fcmfile
        self.start = start
        self.stop = stop
        self.scan_id = scan_id
        self.sbid = sbid
        self.__mfile = None

        if ant_numbers is None:
            ant_numbers = np.arange(NANT) + 1

        self.ant_numbers = ant_numbers

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

    @property
    def metadata_file_name(self):
        if hasattr(self, 'metafilename'):
            return MetadataFile(os.path.join(self.outdir, self.metafilename))
        else:
            raise FileNotFoundError()

    @staticmethod
    def create_from_metafile_and_fcm(metafile, dout, obs_params, fcmfile=None, duration=1*u.hour):
        if isinstance(metafile, str):
            metafile = MetadataFile(metafile)

        sbid = metafile.sbid
        scan_id = metafile.d0['scan_id']
            
        t0 = metafile.times[0]
        nbeams = metafile.nbeam
        targname = metafile.source_name_at_time(t0)

        sources = [metafile.source_at_time(b, t0) for b in range(nbeams)]
        phase_centers = [s['skycoord'] for s in sources]
        sbid = metafile.sbid
        scan_id = metafile.d0['scan_id']
        prep = ScanPrep(targname, sbid, scan_id, phase_centers, fcmfile, dout, t0, t0+duration, ant_numbers=ant_numbers)
        prep.metafilename = 'metafile.json'
        metafile.saveto(os.path.join(dout, prep.metafilename))
        prep.save()
        return prep

    @property
    def metafile(self):
        if self.__mfile is None:
            self.__mfile = MetadataFile(self.metadata_file_name)
            
        return self.__mfile

    @staticmethod
    def load_from_metapath(metapath):
        '''
        Assume someone has given me the full path to the metafile.json
        '''
        assert os.path.basename(metapath) == 'metafile.json'
        rootdir = os.path.dirname(metapath)
        return ScanPrep.load(rootdir)

    def calc_meta_file(self, ibeam):
        '''
        Returns a calc metadata file for the given beam
        '''
        results = self.results_file(ibeam)
        mfstub = self.metadata_file
        calc_meta = CalcMetafile(mfstub.data, results)
        return calc_meta

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
