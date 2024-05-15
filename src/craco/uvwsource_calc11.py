#!/usr/bin/env python
"""
UVFITS with UVWs calculated form Calc11
Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
import tempfile
import subprocess
from craft import calc11
from astropy.time import Time
from astropy.coordinates import SkyCoord
from scipy import constants
import pathlib
from collections import OrderedDict
from craft.uvfits import UvFits


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class UvwSourceCalc11:
    '''
    UVFITs file that patches in UVWs from a calc11 run rather than the metadata file 
    or the UVFITS file itself. this will fill in a template calc file with thegiven arguments uisng craft.calc11,
    then execute 'difxcalc' (which needs to be in the path) and parse the results, all in the __init__
    Once the results are available, call uvw_array_at_time() to get the uvw array
    '''
    def __init__(self, ant_positions, target_name:str, target_skycoord:SkyCoord, start_time:Time, end_time:Time, rundir=None):
        '''
        Create a Calc11UVWSource
        @param ant_positions: Dictionary keyed by antenna name. values should be a 3 tuple (or np array) or X,Y,Z IERS position in meters
        @param target_name: string to put into calc file
        @param target_skycoord: SkyCoord of target
        @param start_time: Time to start UVW calculation
        @param end_time: Time to end UVW calcualtion
        @rundir string path to directory to run calc in. If None it will create a temporary directry. 
        '''
        if rundir is None:
            tempdir = tempfile.TemporaryDirectory(prefix=f'calcfile_{target_name}')
            rundir = tempdir.name

        self.rundir = rundir
        os.makedirs(self.rundir, exist_ok=True)
        assert len(ant_positions) >= 1, 'Not enough antennas'
        self.ant_positions = ant_positions
        
        assert target_name is not None, 'No target name specified'
        self.target_name = target_name

        assert target_skycoord is not None, 'No target skycoord specified'
        self.target_skycoord = target_skycoord

        assert end_time > start_time, 'end time must be after start time'
        self.start_time = start_time
        self.end_time = end_time

        self._write_calcfile()
        self._calc_results = self._execute()

    @staticmethod
    def from_uvfits(uvfits_file:UvFits, rundir=None):
        '''
        Populates this UVW source with properties based on the given uvfits_file
        '''
        f = uvfits_file
        c = UvwSourceCalc11(f.antenna_positions,
                        f.target_name,
                        f.target_skycoord,
                        f.tstart,
                        f.tend,
                        rundir)
        return c

    @property
    def results_file_name(self):
        '''
        Name of results file (may not exist)
        '''
        return os.path.join(self.rundir, 'target.im')
    @property
    def calc_file_name(self):
        '''
        Name of input calc file (shoudl exist after init)
        '''
        return os.path.join(self.rundir, 'target.calc')
    
    @property
    def calc_results(self):
        '''
        Parsed calc results
        '''
        return self._calc_results
    
    def _execute(self):
        assert os.path.exists(self.calc_file_name)
        cmd = ['difxcalc', self.calc_file_name]
        myenv = os.environ.copy()
        try:
            os.remove(self.results_file_name)
        except FileNotFoundError:
            pass
        
        subprocess.check_call(cmd, env=myenv)
        assert os.path.exists(self.results_file_name)
        results = calc11.ResultsFile(self.results_file_name)
        return results

    def _write_calcfile(self):
         # Convert antenna table to something that calc11 file can ingest        
    
        start = self.start_time
        stop = self.end_time
        targname = self.target_name
        skycoord = self.target_skycoord
        calcfile = calc11.CalcFile.from_time_src(start, stop, skycoord, targname)
        calcfile.add_antennas(tuple(self.ant_positions.items()))
        calcfile.add_eops()
        calcfile.writeto(self.calc_file_name)
        log.debug('Wrote calcfile to %s', self.calc_file_name)

    def uvw_array_at_time(self, tuvw:Time):
        '''
        Returns interpolated UVW array.
        For all baselines - irresective of flagging
        :returns: [NANT, 3] numpy array in units of seconds
        '''

        #uvw = self.meta_file.uvw_at_time(tuvw)[:, beamid, :] / constants.c
        #d = self._calc_results.scans[0].eval_src0_poly((mjd + offset).utc.value)
        # TODO: THIS IS COPIED FROM CALC_METAFILE.PY - WE PROBABLY SHOUDL REFACTOR THIS

        nant = 36
        uvw = np.zeros((nant,3))
        
        assert len(self._calc_results.telnames) == nant,f'Unexpected nant {nant} {len(self._calc_results.telnames)}'
        nant = len(self._calc_results.telnames)

        offset = 0
        d = self._calc_results.scans[0].eval_src0_poly((tuvw + offset).utc.mjd)
                
        for iant, ant in enumerate(self._calc_results.telnames):
            auvw = d[ant]
            # I dont' know why calc and the metadata file disagree by a minus
            # sign but they do
            uvw[iant, 0] = -auvw['U (m)']
            uvw[iant, 1] = -auvw['V (m)']
            uvw[iant, 2] = -auvw['W (m)']

        # we want seconds, so we take m and divide by c
        uvw_sec = uvw / constants.c

        return uvw_sec

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
