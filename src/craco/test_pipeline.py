#!/usr/bin/env python
import numpy as np
from pylab import *
import os
import pyxrt
from craco_testing.pyxrtutil import *
import time
import pickle

from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import load_plan
from craco import search_pipeline

from craft import uvfits
from craft import craco

import pytest

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
    p = PipelinePlan(f, '')
    return p

@pytest.fixture(scope='module')
def pipeline(plan, xclbin):
    device = pyxrt.device(0)
    xbin = pyxrt.xclbin(xclbin)
    uuid = device.load_xclbin(xbin)
    # Create a pipeline 
    p = search_pipeline.Pipeline(device, xbin, plan)
    return p


def test_iplist(pipeline, plan):
    pline = pipeline
    print('***** listing IPs *****')
    for ip in pline.xbin.get_ips():
        print(ip.get_name())


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
