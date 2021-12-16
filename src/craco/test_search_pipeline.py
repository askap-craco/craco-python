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

from craco.search_pipeline import waitall, print_candidates, location2pix, NBINARY_POINT_THRESHOLD, NBINARY_POINT_FDMTIN, merge_candidates_width_time

from craft import uvfits
from craft import craco

import pytest
    #candidate_dtype = [('snr', np.int16),
    #('loc_2dfft', np.uint16),
    #               ('boxc_width', np.uint8),
    #               ('time', np.uint8),
    #               ('dm', np.uint16)]


def test_merge_cands_no_merge():
    '''Check you get back what you asked for when merging candidates with no overlaps
    '''
    c = np.array([(576, 0, 0, 6, 0),
                  (576, 0, 0, 6, 1),
                  (576, 0, 0, 6, 2),
                  (576, 0, 0, 6, 3)],
                 dtype=search_pipeline.candidate_dtype)

    cout = np.array(list(merge_candidates_width_time(c)))

    assert np.all(cout == c)

def test_merge_cands_with_merge():
    '''
    Check you get a merge with an example from HW testing
    '''

    cextra = np.array([
        (576, 0, 0, 7, 0),
          (522, 0, 1, 8, 0),
          (576, 0, 0, 7, 1),
          (522, 0, 1, 8, 1),
          (576, 0, 0, 7, 2),
          (522, 0, 1, 8, 2),
          (576, 0, 0, 7, 3),
          (522, 0, 1, 8, 3)],
                       dtype=search_pipeline.candidate_dtype)


    cout = np.array(list(merge_candidates_width_time(cextra)))

    c = np.array([(576, 0, 0, 7, 0),
                  (576, 0, 0, 7, 1),
                  (576, 0, 0, 7, 2),
                  (576, 0, 0, 7, 3)],
                dtype=search_pipeline.candidate_dtype)

    print('c', c)
    print('cout', cout)

    assert np.all(cout == c)

def test_merge_cands_out_of_window():
    '''
    Check you get a merge with an example from HW testing
    '''

    cextra = np.array([
        (576, 0, 0, 7, 0),
          (522, 0, 1, 9, 0),
          (576, 0, 0, 7, 1),
          (522, 0, 1, 9, 1),
          (576, 0, 0, 7, 2),
          (522, 0, 1, 9, 2),
          (576, 0, 0, 7, 3),
          (522, 0, 1, 9, 3)],
                       dtype=search_pipeline.candidate_dtype)


    cout = np.array(list(merge_candidates_width_time(cextra)))
    assert np.all(cout == cextra)




