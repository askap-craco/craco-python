#!/usr/bin/env python
import numpy as np
import pylab
import os
import pyxrt
from craco_testing.pyxrtutil import *
import time
import pickle
import copy

from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import load_plan
from craco import search_pipeline

from craco.search_pipeline import waitall, print_candidates, location2pix, NBINARY_POINT_THRESHOLD, NBINARY_POINT_FDMTIN, merge_candidates_width_time

from craft import uvfits
from craft import craco

import pytest
NTBLK = 11

__author__ = 'Keith Bannister <keith.bannister@csiro.au>'

@pytest.fixture(scope='module')
def fitsname():
    #return '/data/craco/ban115/test_data/nant30/frb_d0_t0_a1_sninf_lm00/frb_d0_t0_a1_sninf_lm00.fits'
    return '/data/craco/ban115/craco-python/src/craco/frb_d0_lm0_nt16_nant24.fits'

@pytest.fixture(scope='module')
def xclbin():
    return '/data/craco/ban115/builds/binary_container_01482863.xclbin'

@pytest.fixture(scope='module')
def plan(fitsname):
    f = uvfits.open(fitsname)
    p = PipelinePlan(f, '--threshold 0 --ndm 4')
    return p

@pytest.fixture(scope='module')
def pipeline(plan, xclbin):
    device = pyxrt.device(0)
    xbin = pyxrt.xclbin(xclbin)
    uuid = device.load_xclbin(xbin)
    # Create a pipeline 
    p = search_pipeline.Pipeline(device, xbin, plan, alloc_device_only_buffers=True)
    return p


def test_iplist(pipeline, plan):
    '''
    CHeck we can get a list of IPs
    Dont care what they are as they names will change and I don't mind if they do
    '''
    print('***** listing IPs *****')
    for ip in pipeline.xbin.get_ips():
        print(ip.get_name())

def test_mainbuf_address_cover_everything(pipeline, plan):
    addresses = np.array([b.buf.address() for b in pipeline.all_mainbufs])
    print(addresses)
    b = pipeline.all_mainbufs[0]
    print(b.nparr.size, b.nparr.itemsize)
    assert np.all(addresses[1:] - addresses[:-1] == b.nparr.size*b.nparr.itemsize)

def test_imaging_dm0_data0(pipeline, plan):
    ''' 
    Make sure clearing the input makes the output all zeros
    This hasn't worked at some point  tblk=0 returns candidates above threshold=0
    which is officially bloody infuriating
    '''
    pipeline.candidates.nparr['snr'] = -1
    pipeline.candidates.copy_to_device()
    
    # reset boxcar history
    pipeline.boxcar_history.clear()
    for b in pipeline.all_mainbufs:
        b.clear()
        
    values = copy.deepcopy(plan.values)

    # clearing by runnign FDMT does work. Clearing by clearing main buffer doesn't work
    values.run_fdmt = True
    values.run_image = False
    pipeline.inbuf.clear()
    for tblk in range(NTBLK):
        starts = pipeline.run(tblk, values)
    
    values.run_fdmt = False
    values.run_image = True
    values.threshold = 0.0
    all_cands = []

    for tblk in range(NTBLK):
        starts = pipeline.run(tblk, values)
        waitall(starts)
        orig_cands = pipeline.get_candidates()
        #print_candidates(orig_cands, plan.npix)
        all_cands.append(orig_cands)


    cand_sizes = np.array(list(map(len, all_cands)))
    print('Candidates vs tblk', cand_sizes)
    np.save('imaging_dm0_data0_cands', all_cands)
    assert np.all(cand_sizes == 0)


def test_imaging_dm0_allt_tblk0(pipeline, plan):

    tblk = 0
    # I'm not sure I can calculate expected S/N for a given input level from first principles yet - so we'll
    # hard code it for now
    # input_level = 1<<8 gives an expected snr of 9.0
    input_level = (1<<9)
    expected_snr = 9.0
    loc2pix = np.vectorize(lambda loc: location2pix(loc, plan.npix))
    all_candidates = []
    check = False
    plot = True
    
    for t in [0,7,8, 15]:
        pipeline.candidates.nparr['snr'] = -1
        pipeline.candidates.copy_to_device()

        # reset boxcar history
        pipeline.boxcar_history.nparr[:] = 0
        pipeline.boxcar_history.copy_to_device()
        for b in pipeline.all_mainbufs:
            b.nparr[:] = 0
            #b.nparr[:, 0, tblk, t, :, 0] = input_level
            #(self.plan.nuvrest, self.plan.ndout, NBLK, self.plan.nt, self.plan.nuvwide, 2)
            b.copy_to_device()


        values = copy.deepcopy(plan.values)
        values.run_fdmt = False
        values.run_image = True
        values.threshold = 8.0
        starts = pipeline.run(0, values)
        waitall(starts)
        orig_cands = pipeline.get_candidates()
        print('time offset', t)
        print('Original candidates before merging')
        print(orig_cands)
        print_candidates(orig_cands, plan.npix)
        cands = np.array(list(merge_candidates_width_time(orig_cands)))
        print(f'merged_cands from {len(orig_cands)} to {len(cands)}')
        print_candidates(cands, plan.npix)
        all_candidates.extend(cands)

        if check:
            assert len(cands) > 0, 'Expected candidates!'
            assert len(cands) == plan.nd, 'Expected 1 candidate per DM above the threshold'
            assert np.all(cands['snr'].astype(float)/(1<<NBINARY_POINT_THRESHOLD) == expected_snr)
            assert np.all(loc2pix(cands['loc_2dfft'] == (128, 128)))
            assert np.all(cands['time'] == t)
            assert np.all(cands['dm'] == np.arange(plan.nd))

    all_candidates = np.array(all_candidates)
    np.save('dm0_cands.npy', all_candidates)

def _main():
    parser = get_parser()
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print(f'Values={values}')

    iplist = xbin.get_ips()
    for ip in iplist:
        print(ip.get_name())

    # Create a plan

    f = uvfits.open(values.uv)
    plan = PipelinePlan(f, values)

    # Create a pipeline 
    p = Pipeline(device, xbin, plan)

    fast_baseline2uv = craco.FastBaseline2Uv(plan, conjugate_lower_uvs=True)
    uv_shape     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide)
    uv_shape2     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide, 2)
    uv_out  = np.zeros(uv_shape, dtype=np.complex64)
    uv_out_fixed = np.zeros(uv_shape2, dtype=np.int16)

    if values.wait:
        input('Press any key to continue...')

    logging.info('Clearing data NBLK=%s', NBLK)

#    for ii in range(NBLK):
#        starts = run(p, ii, values)
#        waitall(starts)#

#    logging.info('done clearing')
#    p.mainbuf.copy_from_device()
#    np.save('mainbuf_after_clearing.npy', p.mainbuf.nparr)

    
        
    for iblk, input_data in enumerate(f.time_blocks(plan.nt)):
        if iblk >= values.nblocks:
            break

        print(iblk)
        
        input_flat = craco.bl2array(input_data)
        fast_baseline2uv(input_flat, uv_out)
        np.save(f'uv_data_blk{iblk}.npy', uv_out)

        p.inbuf.nparr[:,:,:,:,0] = np.round(uv_out[:,:,:,:].real*(float(1<<NBINARY_POINT_FDMTIN)))
        p.inbuf.nparr[:,:,:,:,1] = np.round(uv_out[:,:,:,:].imag*(float(1<<NBINARY_POINT_FDMTIN)))
        p.inbuf.copy_to_device()
        
        # Now we need to use baselines data    
        call_start = time.perf_counter()
        starts = run(p, iblk, values)
        wait_start = time.perf_counter()
    
        for istart, start in enumerate(starts):
            print(f'Waiting for istart={istart} start={start}')
            start.wait(0)

            wait_end = time.perf_counter()
            print(f'Call: {wait_start - call_start} Wait:{wait_end - wait_start}: Total:{wait_end - call_start}')

    f.close()
    
    #p.mainbuf.copy_from_device()
    #print(p.mainbuf.nparr.shape)

    p.candidates.copy_from_device()
    p.boxcar_history.copy_from_device()
    print(np.all(p.boxcar_history.nparr == 0))

    p.fdmt_hist_buf.copy_to_device()
    print('inbuf', hex(p.inbuf.buf.address()))
    print('histbuf', hex(p.fdmt_hist_buf.buf.address()))
    print('fdmt_config_buf', hex(p.fdmt_config_buf.buf.address()))

    # Copy data from device
    p.candidates.copy_from_device()
    candidates = p.candidates.nparr[:]

    for ib, mainbuf in enumerate(p.all_mainbufs):
        mainbuf.copy_from_device()
        np.save(f'mainbuf_after_run_b{ib}.npy', mainbuf.nparr)


    ## Find first zero output
    try:
        last_candidate_index = np.where(candidates['snr'] == 0)[0][0]
    except:
        last_candidate_index = len(candidates)

    candidates = candidates[0:last_candidate_index]
    print_candidates(candidates, p.plan.npix)
    np.save('candidates.npy', p.candidates.nparr[:last_candidate_index])
                     
if __name__ == '__main__':
    _main()
