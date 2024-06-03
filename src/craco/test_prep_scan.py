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
import pytest
from craco.prep_scan import ScanPrep
from craft.craco import bl2ant,bl2array,uvw_to_array
from astropy.coordinates import SkyCoord
from astropy.time import Time
import tempfile
from scipy import constants

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

epics_bat = '0x00124624c938c4be'
epics_uvw = '-3.55009e+06 2.67662e+06 4.56763e+06 -3.67115e+06 2.63211e+06 4.49726e+06 -3.45059e+06 2.80125e+06 4.56915e+06 -3.57491e+06 2.75893e+06 4.49878e+06 -3.52723e+06 2.59099e+06 4.63423e+06 -3.64332e+06 2.553e+06 4.56498e+06 -3.7611e+06 2.50663e+06 4.49462e+06 -3.88032e+06 2.45147e+06 4.42313e+06 -3.79402e+06 2.5785e+06 4.42577e+06 -3.70143e+06 2.70724e+06 4.4273e+06 -3.60197e+06 2.8376e+06 4.4277e+06 -3.47181e+06 2.88697e+06 4.49919e+06 -3.3443e+06 2.9267e+06 4.56955e+06 -3.21964e+06 2.95739e+06 4.6388e+06 -3.3287e+06 2.83473e+06 4.63839e+06 -3.43107e+06 2.71254e+06 4.63687e+06 -3.50607e+06 2.50227e+06 4.6986e+06 -3.61764e+06 2.47025e+06 4.63047e+06 -3.73077e+06 2.43049e+06 4.56122e+06 -3.84527e+06 2.3826e+06 4.49086e+06 -3.96088e+06 2.32621e+06 4.41937e+06 -4.07731e+06 2.26094e+06 4.34677e+06 -4.00068e+06 2.38708e+06 4.35053e+06 -3.91839e+06 2.51531e+06 4.35317e+06 -3.82989e+06 2.64562e+06 4.35469e+06 -3.73454e+06 2.77796e+06 4.3551e+06 -3.63165e+06 2.91225e+06 4.35438e+06 -3.49498e+06 2.96944e+06 4.42698e+06 -3.36124e+06 3.01605e+06 4.49847e+06 -3.23065e+06 3.05276e+06 4.56884e+06 -3.10337e+06 3.08026e+06 4.63808e+06 -2.9795e+06 3.09917e+06 4.70621e+06 -3.09796e+06 2.97962e+06 4.70693e+06 -3.20939e+06 2.85991e+06 4.70652e+06 -3.31426e+06 2.74031e+06 4.705e+06 -3.41301e+06 2.62104e+06 4.70236e+06'
metafile = 'testdata/SB053972/SB53972.json.gz'

def test_save_and_load():
    nbeam = 36
    ra = np.arange(nbeam)
    dec = np.arange(nbeam)*0.5 - 30
    beam_coords = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs', equinox='J2000')
    targname = 'SomeFancyFrbField'
    fcm = 'testdata/fcm_20220714.txt'
    outdir = tempfile.mkdtemp(suffix='prep_scan_pytest')
    print(f'Writing to {outdir}')

    start = Time('2023-09-19 00:00:00')
    stop = Time('2023-09-19 01:00:00')
    sbid = 10001
    scan_id=1
    prep = ScanPrep(targname, sbid, scan_id, outdir, fcm)
    prep.add_calc11_configuration(beam_coords, start, stop)
    prep.save()

    prep2 = ScanPrep.load(outdir)
    assert prep.outdir == prep2.outdir
    assert prep.targname == prep2.targname
    assert np.all(prep.beam_phase_centers == prep2.beam_phase_centers)
    assert prep.fcmfile == prep2.fcmfile
    assert prep.start == prep2.start
    assert prep.stop == prep2.stop

def test_create_from_metadata():
    ant_numbers = np.arange(36)+1
    prep = ScanPrep.create_from_metafile(metafile, valid_ant_numbers=ant_numbers)

def test_uvmax_problem():
    '''
    See CRACO-249
    from teh log file:
    2024-05-21 11:43:13,363 [skadi-14:3823758] r186 search_pipeline_sink Returning baselines for iblk=0 start_fid=549566528 fid_mid=549574720.0 mjd_mid=60451.15603319182 tstart=60451.156028071826
    2024-05-21 11:43:10,163 [skadi-03:3825540] r143 mpipipeline Valid ants: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

    '''
    mjd_mid=60451.15603319182
    indir = '/data/craco/craco/SB062502/scans/00/20240521034236'
    indirstart = '/data/craco/craco/SB062502/scans/00/20240521020713'
    beamid = 31
    tmid = Time(mjd_mid, scale='tai', format='mjd')
    prep = ScanPrep.load(indir)
    prep0 = ScanPrep.load(indirstart) # first scan in teh SB
    
    # I have a terrible feeling tstart hasn't changed
    assert indir != indirstart
    assert prep.start > prep0.start, 'Different scans should have differnt start times'

    expected_valid_ants_0based = np.array(' 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24'.split()).astype(int) - 1
    valid_ants_0based = np.array(prep.valid_ant_numbers) - 1
    md = prep.calc_meta_file(beamid)
    baselines = md.baselines_at_time(tmid, expected_valid_ants_0based, beamid)
    print(baselines)
    for blid, uvw in baselines.items():
        a1,a2 = bl2ant(blid)
        u,v,w = uvw_to_array(uvw)*constants.c
        r = np.sqrt(u**2 + v**2)
        print(f'{a1}-{a2} {u}/{v}/{w} r={r}')
        if r >= 6e3:
            print('eh?')
        #assert r <= 6e3




    

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
