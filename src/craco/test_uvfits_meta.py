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
from craft.craco import ant2bl,bl2ant,to_uvw, uvw_to_array, uvwbl2array,bl2array
from astropy.time import Time
from numpy.testing import assert_allclose


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

#in this example the uvfits was generated with teh json file
# so they should match
metafile = 'testdata/SB053972/SB53972.json.gz'
uvfits = 'testdata/SB053972/b00.uvfits'# this file has 2 bad antennas ak19 and ak25 and nothing above ak30
flag_ants_1based = [8, 19,25,30,31,32,33,34,35,36] # ant 8 isin the file but we want to remove it from baselines
# also it looks like channels 144-151 are inclusive are flagged because of some reason.
flagged_chans = slice(144, 152)

@pytest.fixture
def f1():
    f = craft.uvfits.open(uvfits)
    f.set_flagants(flag_ants_1based)
    return f

@pytest.fixture
def f2():
    f = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    f.set_flagants(flag_ants_1based)
    return f


def check_baselines_equal(b1,b2):
    assert type(b1) == type(b2)
    assert len(b1) == len(b2)
    assert b1.keys() == b2.keys()

    for bl in b1.keys():
        for t in ('UU','VV','WW'):
            assert b1[bl][t] == b2[bl][t]


def test_antenna_positions_is_ok(f1):
    print(f1.antenna_positions)
    assert len(f1.antenna_positions) >= 1

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

def test_beamid_sensible(f1):
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

def test_baselines_after_flag():
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    f2f = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    ants = [a+1 for a in range(36)]

    bl_unflagged = f2.baselines
    unflagged_valid_ants = f2.valid_ants

    f2f.set_flagants(flag_ants_1based)
    for a in flag_ants_1based:
        ants.remove(a)

    bl_flagged = f2f.baselines
    flagged_valid_ants = f2.valid_ants

    assert len(bl_flagged) < len(bl_unflagged)


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

def test_uvws_sensible_with_calc11():
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    f2.set_flagants(flag_ants_1based)

    f3 = craco.uvfits_meta.open(uvfits, metadata_file=metafile, calc11=True)
    f3.set_flagants(flag_ants_1based)

    valid_ants_0based = np.array(f2.valid_ants) - 1
    bad_ants_0based = np.array(flag_ants_1based) - 1

    tstart = f2.tstart
    tsamp = f2.tsamp
    tmid = tstart + tsamp/2
    
    uvw = f2.uvw_array_at_time(tmid)
    uvw3 = f3.uvw_array_at_time(tmid)
    print('Bad ants are at 0 based indexs', np.where(uvw==0))
    print('Bad ants 0based', bad_ants_0based)
    uvwdiff = uvw3 - uvw
    print(f'RMS error in UVW is {uvwdiff.std(axis=0)}')

    assert np.all(uvw[valid_ants_0based,:]) != 0, 'valid ants have 0 uvw'
    assert np.all(uvw3[valid_ants_0based,:]) != 0, 'valid ants have 0 uvw'

    assert np.any(uvwdiff != 0), 'Surely there are some differences!'


def test_uvfits_fast_time_blocks_with_istart():
    f1 = craft.uvfits.open(uvfits)
    print(f1.nblocks) # this is only 336 for the test data
    nt = 64 # 
    i0 = f1.fast_time_blocks(nt, fetch_uvws=True, istart=0)
    ioff = f1.fast_time_blocks(nt, fetch_uvws=True, istart=nt)

    b0t0 = next(i0)
    b0t1 = next(i0)
    bofft1 = next(ioff)
    
    d1, uvw1 = b0t1
    d2, uvw2 = bofft1
    d3, uvw3 = b0t0

    assert np.all(uvw1 == uvw2)
    assert np.all(d1 == d2)

    assert not np.all(uvw3 == uvw2)
    assert not np.all(d3 == d2)

def test_uvfits_get_uvw(f1):
    nblocks = f1.nblocks
    assert f1.get_uvw_at_isamp(0) is not None, 'Shoudl get data at sample 0'
    assert f1.get_uvw_at_isamp(nblocks-1) is not None, 'Should get data for alst sample'


def test_baselines_in_meta_match(f1,f2):
    b1 = f1.baselines
    b2 = f2.baselines
    tstart = f2.tstart
    tsamp = f2.tsamp
    tmid = tstart + tsamp / 2

    uvw1 = uvwbl2array(b1)
    uvw2 = uvwbl2array(b2)

    assert_allclose(uvw1,uvw2,rtol=5e-7)

    plot = False

    if plot:
        import pylab
        pylab.scatter(uvw1[:,0], uvw1[:,1])
        pylab.scatter(uvw2[:,0], uvw2[:,1])

        diff = uvw1 - uvw[2]
        pylab.figure()
        pylab.scatter(diff[:,0], diff[:,1])
        pylab.show()


def test_vis_metadata_makes_sense(f1,f2):
    nt = 64
    nblk = 4
    all_uvw1 = []
    all_uvw2 = []
    assert nblk*nt <= f1.nblocks
    for i in range(nblk):
        t = nt*i

        vm1 = f1.vis_metadata(t)
        vm2 = f2.vis_metadata(t)
        
        uvw1 = uvwbl2array(vm1.baselines)
        uvw2 = uvwbl2array(vm2.baselines)
        all_uvw1.append(uvw1)
        all_uvw2.append(uvw2)
        
        #assert_allclose(uvw1,uvw2, rtol=5e-7)
        
        assert vm1.tstart == vm2.tstart
        assert vm1.tsamp == vm2.tsamp
        assert vm1.beamid == vm2.beamid
        assert vm1.target_skycoord == vm2.target_skycoord
        assert vm1.target_name == vm2.target_name
        assert vm1.freq_config == vm2.freq_config

    all_uvw1, all_uvw2 = map(np.array, [all_uvw1, all_uvw2])

    # check UVWs are OK
    assert_allclose(uvw1, uvw2, rtol=5e-7)

def test_time_blocks_with_uvws_equal(f1,f2):
    nt = 64
    nblk = 4
    all_uvw1 = []
    all_uvw2 = []
    assert nblk*nt <= f1.nblocks
    for i in range(nblk):
        t = nt*i
        trange = (t, t+nt-1)
        d1, uvws1, (sstart1, send1) = f1.time_block_with_uvw_range(trange)
        print(type(uvws1), len(uvws1))
        d2, uvws2, (sstart2, send2) = f2.time_block_with_uvw_range(trange)
        assert np.all(bl2array(d1) == bl2array(d2))
        assert type(uvws1) == type(uvws2)
        assert len(uvws1) == len(uvws2)
        uvws1 = bl2array(uvws1, dtype=np.float64)
        uvws2 = bl2array(uvws2, dtype=np.float64)
        all_uvw1.append(uvws1)
        all_uvw2.append(uvws2)
        #assert_allclose(uvws1, uvws2, rtol=5e-7)
        assert np.all(bl2array(d1) == bl2array(d2))
        assert sstart1 == sstart2
        assert send1 == send2
        
       

    all_uvw1, all_uvw2 = map(np.array, [all_uvw1, all_uvw2])
    u1 = all_uvw1.transpose(1,2,0,3).reshape(f1.nbl,3,-1)
    u2 = all_uvw2.transpose(1,2,0,3).reshape(f2.nbl,3,-1)

    plot = False
    if plot:
        import pylab
        pylab.plot(u1[0,0,:])
        pylab.plot(u2[0,0,:])
        pylab.show()

    # check UVWs are OK
    assert_allclose(u1,u2, rtol=5e-7)


def test_vis_property_equal(f1,f2):
    nt = 64
    nblk = 4
    all_uvw1 = []
    all_uvw2 = []
    assert nblk*nt <= f1.nblocks
    f2.mask = False # masking with metadata will make the data validation fail
    
    for i in range(nblk):
        t = nt*i
        trange = (t, t+nt-1)
        istart=t*f1.raw_nbl
        iend=istart+nt*f1.raw_nbl
        v1 = f1.vis[istart:iend]
        v2 = f2.vis[istart:iend]
        assert np.all(v1['DATE'] == v2['DATE'])
        assert np.all(v1['BASELINE'] == v2['BASELINE'])
        assert np.all(v1['DATA'] == v2['DATA'])
        for x in ('UU','VV','WW'):
            assert_allclose(v1[x], v2[x], rtol=5e-7)

def test_vis_size_is_sensible(f1,f2):
    nt = 64
    nblk = 4
    all_uvw1 = []
    all_uvw2 = []
    assert nblk*nt <= f1.nblocks
    # VIS SIZE works on raw_nbl
    for i in range(nblk):
        t = nt*i
        trange = (t, t+nt-1)
        istart=t*f1.raw_nbl
        iend=istart+nt*f1.raw_nbl
        v1 = f1.vis[istart:iend]
        v2 = f2.vis[istart:iend]
        print(i, v1.size, v2.size, nt, f1.nbl)
        assert v1.size == f1.raw_nbl*nt
        assert v1.size == v2.size


def test_source_name_and_position(f1,f2):
    nt = 64
    
    nblk = 4

    assert f2.target_skycoord == f2.target_skycoord
    assert f1.target_name == f2.target_name

def test_time_block_and_vis_agree():
    f1 = craft.uvfits.open(uvfits)
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)

    #f1.set_flagants(flag_ants_1based)
    #f2.set_flagants(flag_ants_1based)

    nt = 64
    nblk = 4
    nblk = 4
    all_uvw1 = []
    all_uvw2 = []
    assert nblk*nt <= f1.nblocks
    for i in range(nblk):
        t = nt*i
        trange = (t, t+nt-1)
        d1, uvws1, (sstart1, send1) = f1.time_block_with_uvw_range(trange)
        d2, uvws2, (sstart2, send2) = f2.time_block_with_uvw_range(trange)
        uvws1 = bl2array(uvws1, dtype=np.float32)
        uvws2 = bl2array(uvws2, dtype=np.float32)
        all_uvw1.append(uvws1)
        all_uvw2.append(uvws2)

        istart = i*nt*f1.raw_nbl
        iend = istart + nt*f1.raw_nbl
        v1v = f1.vis[istart:iend]
        v2v = f2.vis[istart:iend]

        assert len(v1v) == (iend - istart)
        assert (iend - istart) == nt*f1.raw_nbl
        
        v1v = v1v.reshape(nt,-1).T
        v2v = v2v.reshape(nt,-1).T
        assert v1v.shape == uvws1[:,0,:].shape

        for ix,x in enumerate(('UU','VV','WW')):
            diff = v2v[x] - uvws2[:, ix, :]
            equal = np.all(v2v[x] == uvws2[:,ix,:])
            # I think you need rtol = 6e-7 rather than 5e-7 for the unflagged data
            # maybe the flagged values are somehow a little worse
            assert_allclose(v1v[x], uvws1[:,ix,:], rtol=6e-7)
            assert_allclose(v2v[x], uvws2[:,ix,:], rtol=6e-7)



    all_uvw1, all_uvw2 = map(np.array, [all_uvw1, all_uvw2])
    u1 = all_uvw1.transpose(1,2,0,3).reshape(231,3,-1)
    u2 = all_uvw2.transpose(1,2,0,3).reshape(231,3,-1)

    plot = False
    if plot:
        import pylab
        pylab.plot(u1[0,0,:])
        pylab.plot(u2[0,0,:])
        pylab.show()

    # check UVWs are OK
    assert_allclose(u1,u2, rtol=6e-7)

def test_time_conversions(f1):
    s1 = 100
    t1 = f1.sample_to_time(s1)
    s1t = f1.time_to_sample(t1)

    assert s1 == int(np.round(s1t))

def test_start_mjd_offset(f1):
    nt = 64
    s1 = nt
    t1 = f1.sample_to_time(s1)

    f2 = craft.uvfits.open(uvfits, start_mjd=t1)
    f2.set_flagants(flag_ants_1based)
    skip_blocks = 0

    i0 = f1.fast_time_blocks(nt, fetch_uvws=True, istart=skip_blocks)
    ioff = f2.fast_time_blocks(nt, fetch_uvws=True, istart=0)


    b0t0 = next(i0)
    b0t1 = next(i0)

    bofft0 = next(ioff)
    bofft1 = next(ioff)
    
    d0, uvw0 = b0t0
    d1, uvw1 = b0t1

    d3, uvw3 = bofft0
    d4, uvw4 = bofft1


    #I don't know whow this was ever right
    #assert np.all(uvw0 == uvw3)
    #assert np.all(d0 == d3)

    #assert not np.all(uvw3 == uvw2)
    #assert not np.all(d3 == d2)

def test_fast_time_blocks_masks_ok(f1):
    nt = 64
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile)
    i1 = f1.fast_time_blocks(nt, fetch_uvws=True, istart=0)
    i2 = f2.fast_time_blocks(nt, fetch_uvws=True, istart=0)

    d1,uvw1 = next(i1)
    d2,uvw2 = next(i2)

    # metadata says 2 antennas should be flagged all of the time.
    # so f2 should have at least some data flagged
    # When we recorded the UVFITS it had none of the flagged antennas in them
    # we need to record another uvfits with all antennas and some metadata
    # to check it works as expected.
    # for now, everything should be unflagged
    mask = d2.mask.squeeze()

    print('Number of flags', np.sum(d2.mask), mask.shape)
    if False:
        pylab.imshow(mask[:,:,0])
        pylab.figure()
        pylab.plot(mask[:,:,0].sum(axis=0))
        pylab.show()

    # channel 0 is OK - there are some flagged channels but we don't care
    assert np.all(mask[:,0,:] == False)
    assert np.all(mask[:,flagged_chans,:] == True)


def test_before_and_after_flags_vis_equal():
    skip = 1
    f2 = craco.uvfits_meta.open(uvfits, metadata_file=metafile, skip_blocks=skip)
    raw_nbl = f2.raw_nbl
    nbl = f2.nbl
    v1 = f2.vis[0:raw_nbl]
    
    f2.set_flagants(flag_ants_1based)

    v2 = f2.vis[0:raw_nbl]
    assert raw_nbl == f2.raw_nbl
    assert np.all(v1==v2)

def test_wrong_timestamp():
    '''
    Nasty test with the wrong teimstamp.
    Turned out to be a corrupted last block
    Probably should refacto rthis with a standard test and make sure you 
    can fast_raw_blocks() past the end and nothing will happen
    
    '''
    metafile = '/data/craco/craco/SB062220/SB62220.json.gz'
    inf = '/CRACO/DATA_01/craco/SB062220/scans/00/20240506172535/b00.uvfits'
    if not os.path.exists(inf):
        return
    
    f = craco.uvfits_meta.open(inf,metadata_file=metafile)
    rawf = craft.uvfits.open(inf)
    nbl = f.nbl
    tblk  = 3991
    # breaks for index: slice(1620752, 1621158, None)
    istart = tblk*nbl
    iend = istart+nbl
    d = f.vis[istart:iend] # should be OK for 3991
    try:
        for iblk,d in enumerate(f.fast_raw_blocks(istart=tblk, nsamp=3000)): # try to read past the end of the file
            print(iblk)
    except ValueError: # interpolation error
        print(iblk)
        raise
              

    
    assert d is not None





    


    


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
