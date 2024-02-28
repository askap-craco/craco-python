#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import os
import sys
import logging
import pytest
import json
import astropy.units as u
import shutil

from craft.craco import ant2bl,bl2ant,to_uvw, uvw_to_array, uvwbl2array,bl2array
from craco.metadatafile import MetadataFile

from craco.prep_scan import ScanPrep
from craco.calc_metafile import CalcMetafile
from IPython import embed

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

metafile = './testdata/SB053972/SB53972.json.gz'
fcmfile = 'testdata/fcm_20220714.txt'


@pytest.fixture
def metastub():
    '''
    Creates a sub of a metadata file containing 2 integrations
    Returns a metadata file with just the stub
    '''
    mf = MetadataFile(metafile)
    stubdata = mf.data[16:18]
    stubout = os.path.join(os.path.dirname(metafile), 'stub.json')
    with open(stubout, 'w') as fout:
        json.dump(stubout, stubdata)


@pytest.fixture
def mf():
    return MetadataFile(metafile)


@pytest.fixture
def mfstub(mf):
    return mf[16:18]

def check_source_equal(m1, m2,  t0):
    for beam in range(36):
        s1 = m1.source_at_time(beam, t0)
        s2 = m2.source_at_time(beam, t0)
        for k in ('ra','dec','epoch','skycoord'):
            assert s1[k] == s2[k]


def test_metadata_stub_same(mf, mfstub):
    t0 = mfstub.times[0]
    beam = 0
    assert mf.source_name_at_time(t0) == mfstub.source_name_at_time(t0)
    assert np.all(mf.flags_at_time(t0) == mfstub.flags_at_time(t0))
    assert mf.sbid == mfstub.sbid
    
    assert_allclose(mf.uvw_at_time(t0), mfstub.uvw_at_time(t0))
    check_source_equal(mf, mfstub, t0)

def test_scanprep_and_metafile(mfstub):
    dout = os.path.join('/tmp','scanprep')
    try:
        shutil.rmtree(dout)
    except FileNotFoundError:
        pass

    os.makedirs(dout, exist_ok=True)

    nbeams = 36
    beam = 0
    prep = ScanPrep.create_from_metafile_and_fcm(mfstub, fcmfile, dout, duration=15*u.minute)
    prep2 = ScanPrep.load(dout)
    t0 = mfstub.times[0]
    calc_meta = prep.calc_meta_file(beam)
    calc_meta2 = prep2.calc_meta_file(beam)
    check_source_equal(mfstub, calc_meta, t0)
    check_source_equal(mfstub, calc_meta2, t0)
    uvw1 = mfstub.uvw_at_time(t0, beam)
    uvw2 = calc_meta.uvw_at_time(t0)

    flags = mfstub.flags_at_time(t0)

    nant = len(flags)
    valid_ants_0based = [iant for iant in range(nant) if not flags[iant]]
    bl1 = uvwbl2array(mfstub.baselines_at_time(t0, valid_ants_0based, beam))*3e8
    bl2 = uvwbl2array(calc_meta.baselines_at_time(t0, valid_ants_0based, None))*3e8

    print('Stderr', (bl1 - bl2).flatten().std())
    
    uvw1 = uvw1[~flags, :]
    uvw2 = uvw2[~flags, :]

    stderr = []
    offsets = np.arange(-10,10,0.1)
    for ioff, off in enumerate(offsets):
        blx = uvwbl2array(calc_meta.baselines_at_time(t0+off*u.second, valid_ants_0based, None))*3e8
        stderr.append((blx - bl1).flatten().std())

    plot = True
    if plot:
        pylab.figure()
        pylab.plot(offsets, stderr)
        pylab.xlabel('offset (second)')
        pylab.ylabel('UVW stderr (m)')
        
        pylab.figure()
        pylab.plot(uvw1[:,0], uvw1[:,1],'rx')
        pylab.plot(uvw2[:,0], uvw2[:,1],'go')
        
        pylab.figure()
        pylab.plot(bl1[:,0], bl1[:,1], 'rx')
        pylab.plot(bl2[:,0], bl2[:,1], 'go')
    
        pylab.show()
    #embed()
    
    assert_allclose(bl1,bl2)


    
    
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
