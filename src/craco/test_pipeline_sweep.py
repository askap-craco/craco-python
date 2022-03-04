#!/usr/bin/env python
import numpy as np
import pylab
import os
import pyxrt
from craco_testing.pyxrtutil import *
import time
import pickle
import copy
import logging


from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import get_parser
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


def test_dm0_tsweep_makes_sense(pipeline, plan):
    '''
    Test that DM0 gives candidates with corrrect time offset 
    '''
    dm =0 
    for t in range(512):
        pulseinjector = Fake_injector(dm, t)
        for iblock,block in enumerate(pulseinjector):
            pipeline.copy_input(block)
            pipeline.run(0, values).wait()
            cand = pipeline.get_candidates()
            cand.writeto(f'test_dm0_tsweep_makese_sense_t{t}_blk{iblock}.npy')

            # in an ideal case we'd check that the candidate values are correct
        

        
