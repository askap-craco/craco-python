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
import craco.uvfits_meta
import craft.uvfits
from IPython import embed
from craft.craco import ant2bl,bl2ant
from craco.metadatafile import to_uvw, uvw_to_array
from astropy.time import Time
from numpy.testing import assert_allclose


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

#in this example the uvfits was generated with teh json file
# so they should match
metafile = 'testdata/SB053972/SB53972.json.gz'
uvfits = 'testdata/SB053972/b00.uvfits'# this file has 2 bad antennas ak19 and ak25 and nothing above ak30
flag_ants_1based = [19,25,30,31,32,33,34,35,36]


def check_baselines_equal(b1,b2):
    assert type(b1) == type(b2)
    assert len(b1) == len(b2)
    assert b1.keys() == b2.keys()

    for bl in b1.keys():
        for t in ('UU','VV','WW'):
            assert b1[bl][t] == b2[bl][t]

def test_to_uvw():
    uvout = to_uvw([1.0,2.0,3.0])
    print('UVOUT', uvout)
    assert uvout['UU'] == 1.0
    assert uvout['VV'] == 2.0
    assert uvout['WW'] == 3.0

def test_baseline_order_and_baselines_match():
    f1 = craft.uvfits.open(uvfits)

    # Check I havnt broken anything with the baseline ordering
    # refactoring I did with uvfits - probably shouuld put in 
    b1 = f1.baselines
    b2 = f1._find_baseline_order()
    check_baselines_equal(b1,b2)

def test_beam_from_filename():
    assert craft.uvfits.parse_beam_id_from_filename(uvfits) == 0

def test_beamid_sensible():
    f1 = craft.uvfits.open(uvfits)
    assert f1.beamid == 0

def test_valid_ants_sensible():
    f1 = craft.uvfits.open(uvfits)
    ants = [a+1 for a in range(36)]
    assert f1.valid_ants == ants

def test_valid_ants_after_flag():
    f1 = craft.uvfits.open(uvfits)
    ants = [a+1 for a in range(36)]

    f1.set_flagants(flag_ants_1based)
    for a in flag_ants_1based:
        ants.remove(a)

    assert f1.valid_ants == ants

def test_times_sensible():
    f1 = craft.uvfits.open(uvfits)
    assert 'DATE-OBS' not in f1.header
    #date_obs = Time(f1.header['DATE-OBS'], format='isot', scale='utc')
    first_sample_date = Time(f1.start_date, format='jd', scale='utc')
    print('First sample', first_sample_date.utc.iso, first_sample_date.tai.iso)

def test_uvws_sensible():
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    f2.set_flagants(flag_ants_1based)
    valid_ants_0based = np.array(f2.valid_ants) - 1
    bad_ants_0based = np.array(flag_ants_1based) - 1

    tstart = f2.tstart
    tsamp = f2.tsamp
    tmid = tstart + tsamp/2
    
    uvw = f2.meta_file.uvw_at_time(tmid)[:,0,:]
    print('Bad ants are at 0 based indexs', np.where(uvw==0))
    print('Bad ants 0based', bad_ants_0based)

    assert np.all(uvw[valid_ants_0based,:]) != 0, 'valid ants have 0 uvw'
    
    #assert np.all(uvw[bad_ants_0based,:] == 0), 'Bad ants have nonzero uvw'


def test_baselines_in_meta_match():
    f1 = craft.uvfits.open(uvfits)
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)

    f1.set_flagants(flag_ants_1based)
    f2.set_flagants(flag_ants_1based)

    b1 = f1.baselines
    b2 = f2.baselines
    tstart = f2.tstart
    tsamp = f2.tsamp
    tmid = tstart + tsamp / 2

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for bl in b1.keys():
        u1 = uvw_to_array(b1[bl])
        u2 = uvw_to_array(b2[bl])
        print(bl, bl2ant(bl), u1, u2, u1 - u2)
        x1.append(u1[0])
        x2.append(u2[0])
        y1.append(u1[1])
        y2.append(u2[1])

    x1,x2,y1,y2 = map(np.array, (x1,x2,y1,y2))
    assert_allclose(x1,x2,rtol=5e-7)
    assert_allclose(y1,y2,rtol=5e-7)

    plot = False

    if plot:
        import pylab
        pylab.scatter(x1,y1)
        pylab.scatter(x2,y2)
        
        pylab.figure()
        pylab.scatter(x2-x1,y2-y1)
        pylab.show()
    
    
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
