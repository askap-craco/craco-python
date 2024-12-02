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
from craft.corruvfits import CorrUvFitsFile
from astropy.time import Time
from astropy import units as u
from craco.uvfitsfile_sink import *
from craft.craco import ant2bl, baseline_iter
import craco.card_averager
from craft import uvfits


import craco


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class TestVisblock:
     def __init__(self, d, mjdmid, uvw, valid_ants_0based):
        self.data = d
        self.fid_start = 1234
        self.fid_mid = self.fid_start + np.uint64(NSAMP_PER_FRAME//2)
        self.mjd_mid = mjdmid
        self.uvw = uvw
        self.source_index = 0
        nant = len(valid_ants_0based)
        self.antflags = np.zeros(nant, dtype=bool)
        af = self.antflags
        self.baseline_flags = np.array([af[blinfo.ia1] | af[blinfo.ia2] for blinfo in baseline_iter(valid_ants_0based)])

fcent = 850e6
foff = 1e6
npol = 1
tstart = Time.now().mjd
source_list = [{'name':'test', 'ra':123, 'dec':-33}]
antennas = []
extra_header= {}
nbeam = 36
nant = 10
valid_ants_0based = np.arange(nant)
nc_per_card = 24
nt = 32
npol = 1
vis_fscrunch = 6
vis_tscrunch = 4
real_dtype = np.float32
cplx_dtype = np.float32
nrx = 72
nchan = nc_per_card*nrx // vis_fscrunch
vis_nt = nt // vis_tscrunch
nbl = nant*(nant-1)//2
dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc_per_card, nt, npol, vis_fscrunch, vis_tscrunch, real_dtype, cplx_dtype)
valid_ants_0based = np.arange(nant)
uvw = np.random.randn(nbl*3).reshape(nbl,3)
tstart = Time(60467.28828320785, format='mjd', scale='utc')
fits_sourceidx = 1
inttime = 13.4e-3
mjds = np.array([(tstart + inttime*u.second*i).utc.value for i in range(vis_nt)])
sampleidxs = np.arange(vis_nt)
mjdiffs = sampleidxs*inttime/86400
baseline_info = list(baseline_iter(valid_ants_0based))
blids = [bl.blid for bl in baseline_iter(valid_ants_0based)]


def test_fast_uvfits_is_equivalent_to_slow():        

    input = np.zeros(nrx, dtype=dt)
    input['vis'][:] = np.random.randn(*input['vis'].shape)
    vis_block = TestVisblock(input['vis'], tstart, uvw, valid_ants_0based)
    
    slow_uvout = CorrUvFitsFile('slow.uvfits',
                                fcent,
                                foff,
                                nchan,
                                npol,
                                tstart.value,
                                source_list,
                                antennas,
                                extra_header=extra_header,
                                instrume='CRACO')
    fast_uvout = CorrUvFitsFile('fast.uvfits',
                                fcent,
                                foff,
                                nchan,
                                npol,
                                tstart.value,
                                source_list,
                                antennas,
                                extra_header=extra_header,
                                instrume='CRACO')
    
    (dreshape, weights, uvw_baselines) = prepare_data_slow(vis_block, nchan, npol, baseline_info)
    write_data_slow(slow_uvout, uvw_baselines,dreshape, weights, fits_sourceidx, mjds, blids, inttime)
    slow_uvout.close()
    sz = os.path.getsize('slow.uvfits')

    prepper = DataPrepper(fast_uvout, baseline_info, vis_nt, fits_sourceidx, inttime)
    # write to disk
    prepper.write(vis_block)
    fast_uvout.close()
    szf = os.path.getsize('fast.uvfits')

    assert sz == szf
    slow = uvfits.open('slow.uvfits')
    fast = uvfits.open('fast.uvfits')
    assert slow.vis.size == fast.vis.size
    n = slow.vis.size
    
    keys = ('UU','VV','WW','DATE', 'BASELINE','INTTIM','FREQSEL','SOURCE','INTTIM','DATA')
    for k in keys:
        np.testing.assert_equal(slow.vis[0:n][k] , fast.vis[0:n][k], err_msg=f'Incorrect values for {k}')
        
    print('Data identical!')

def test_uvfitsfilesink_tscrunch_identical():
    '''
    Bleach, how do we do this?
    '''
    tscrunch = 4

    input = np.zeros(nrx, dtype=dt)
    dt_scrunched = craco.card_averager.get_averaged_dtype(nbeam, nant, nc_per_card, nt, npol, vis_fscrunch, vis_tscrunch*tscrunch, real_dtype, cplx_dtype)
    input['vis'][:] = np.random.randn(*input['vis'].shape)
    input['vis'][:] = np.arange(np.prod(input['vis'].shape)).reshape(input['vis'].shape)

    input_scrunched = np.zeros(nrx, dtype=dt_scrunched)
    _, nbl, vis_nc, vis_nt, _ = input['vis'].shape
    np.mean(input['vis'].reshape(nrx, nbl, vis_nc, vis_nt//tscrunch, tscrunch, 2), axis=4, out=input_scrunched['vis'])

    vis_block = TestVisblock(input['vis'], tstart, uvw, valid_ants_0based)
    vis_block_scrunched = TestVisblock(input_scrunched['vis'], tstart, uvw, valid_ants_0based)

    
    tscrunch1 = CorrUvFitsFile('tscrunch1.uvfits',
                                fcent,
                                foff,
                                nchan,
                                npol,
                                tstart.value,
                                source_list,
                                antennas,
                                extra_header=extra_header,
                                instrume='CRACO')
    tscrunch4 = CorrUvFitsFile('tscrunch4.uvfits',
                                fcent,
                                foff,
                                nchan,
                                npol,
                                tstart.value,
                                source_list,
                                antennas,
                                extra_header=extra_header,
                                instrume='CRACO')
    
    prepper = DataPrepper(tscrunch1, baseline_info, vis_nt // tscrunch, fits_sourceidx, inttime*tscrunch)
    # write to disk
    prepper.write(vis_block_scrunched)
    tscrunch1.close()
    sz1 = os.path.getsize('tscrunch1.uvfits')

    prepper4 = DataPrepper(tscrunch4, baseline_info, vis_nt, fits_sourceidx, inttime, tscrunch=4)
    # write to disk
    prepper4.write(vis_block)
    tscrunch4.close()
    sz4 = os.path.getsize('tscrunch4.uvfits')

    assert sz1 == sz4
    slow = uvfits.open('tscrunch1.uvfits')
    fast = uvfits.open('tscrunch4.uvfits')
    assert slow.vis.size == fast.vis.size
    n = slow.vis.size
    
    keys = ('DATA','UU','VV','WW','DATE', 'BASELINE','INTTIM','FREQSEL','SOURCE','INTTIM',)
    for k in keys:
        np.testing.assert_allclose(slow.vis[0:n][k], fast.vis[0:n][k], err_msg=f'Incorrect values for {k}')
        
    print('Data identical!')
            


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
