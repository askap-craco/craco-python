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
from craco import uvfits_meta
from craft import calc11
from astropy.time import Time
from scipy import constants


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class UvfitsCalc11(uvfits_meta.UvfitsMeta):
    '''
    UVFITs file that patches in UVWs from a calc11 run rather than the metadata file 
    or the UVFITS file itself
    '''
    def __init__(self, hdulist, max_nbl=None, mask=True, skip_blocks=0, metadata_file=None, start_mjd=None,end_mjd=None, rundir=None):

        self._calc_results = None # this is a hack. __init__ needs to run before we can run calc
        super().__init__(hdulist, max_nbl, mask, skip_blocks, metadata_file, start_mjd, end_mjd)
        if rundir is None:
            tempdir = tempfile.TemporaryDirectory(prefix=f'calcfile_beam{self.beamid}')
            rundir = tempdir.name

        self.rundir = rundir
        self._write_calcfile()
        self._calc_results = self._execute()

    @property
    def results_file_name(self):
        return os.path.join(self.rundir, 'target.im')
    @property
    def calc_file_name(self):
        return os.path.join(self.rundir, 'target.calc')
    

    def _execute(self):
        assert os.path.exists(self.calc_file_name)
        cmd = ['/home/ban115/bin/difxcalc', self.calc_file_name]
        myenv = os.environ.copy()
        subprocess.check_call(cmd, env=myenv)
        assert os.path.exists(self.results_file_name)
        results = calc11.ResultsFile(self.results_file_name)
        return results


    def _write_calcfile(self):
        # This is horrific. One day we'll fix it but not today
        # ant_data, antnos = fcm2calc.load_parset(self.fcmfile)

        # Convert antenna table to something that calc11 file can ingest        
        anttab = self.hdulist['AIPS AN']
        antdata = {}
        nant = len(anttab.data)
        antnos = [iant for iant in range(1, nant+1)]
        for iant, row in enumerate(anttab.data):
            pos_str = ','.join(map(str, row['STABXYZ']))
            antdata[iant+1] = {'location.itrf':pos_str,
                               'name':row['ANNAME']}

        start = self.tstart
        stop = self.tend
        targname = self.target_name
        skycoord = self.target_skycoord
        calcfile = calc11.CalcFile.from_time_src(start, stop, skycoord, targname)
        calcfile.add_antdata(antdata, antnos)
        calcfile.add_eops()
        calcfile.writeto(self.calc_file_name)
        log.debug('Wrote calcfile to %s'% self.calc_file_name)

    def uvw_array_at_time(self, tuvw:Time):
        '''
        Returns interpolated UVW array.
        For all baselines - irresective of flagging
        for the beamid returned by self.beamid
        :returns: [NANT, 3] numpy array in units of seconds
        '''


        beamid = self.beamid
        #uvw = self.meta_file.uvw_at_time(tuvw)[:, beamid, :] / constants.c
        #d = self._calc_results.scans[0].eval_src0_poly((mjd + offset).utc.value)
        # TODO: THIS IS COPIED FROM CALC_METAFILE.PY - WE PROBABLY SHOUDL REFACTOR THIS

        nant = 36
        uvw = np.zeros((nant,3))
        # THIS IS A TOTAL BROKEN SMELL! __init__ needs to run before we populate
        # the fields to run calc, but if uvw_array_at_time() gets called to _find_baseline_order()
        # which wants to patch in the UVWs - so the whole thing is broken
        # Anyway, if _calc_results is None, we just return empty data and work it out afterward
        if self._calc_results is None:
            return uvw
        
        assert len(self._calc_results.telnames) == nant

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
