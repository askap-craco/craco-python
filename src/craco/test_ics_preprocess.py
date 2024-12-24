import logging
import numpy as np
import os
from craco.ics_preprocess import get_ics_masks


def test_get_ics_masks_zeros():
    nf = 288 * 3
    nt = 256

    fake_ics_data = np.zeros((nf, nt), dtype=np.float64)
    tf_weights = np.zeros((nf, nt), dtype='bool')

    get_ics_masks(fake_ics_data, tf_weights)

    assert np.all(tf_weights == True)



def test_get_ics_masks_ones():
    nf = 288 * 3
    nt = 256

    fake_ics_data = np.ones((nf, nt), dtype=np.float64)
    tf_weights = np.zeros((nf, nt), dtype='bool')

    get_ics_masks(fake_ics_data, tf_weights)

    assert np.all(tf_weights == True)



def test_get_ics_masks_random():
    nf = 288 * 3
    nt = 256

    fake_ics_data = np.random.normal(0, 1, nf*nt).reshape((nf, nt))
    tf_weights = np.zeros((nf, nt), dtype='bool')

    get_ics_masks(fake_ics_data, tf_weights)

    assert np.all(tf_weights == True)


def test_get_ics_masks_random_RFI():
    nf = 288 * 3
    nt = 256

    fake_ics_data = np.random.normal(0, 1, nf*nt).reshape((nf, nt))
    bad_time_sample = 123

    fake_ics_data[:, bad_time_sample] += 20 
    tf_weights = np.zeros((nf, nt), dtype='bool')

    get_ics_masks(fake_ics_data, tf_weights)

    assert np.all(tf_weights[:, bad_time_sample] == False)

    assert np.all(tf_weights[:, ~bad_time_sample] == True)



