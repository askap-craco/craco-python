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
from craco.candidate_writer import CandidateWriter
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from craft import craco_wcs
from pytest import fixture
from numpy.testing import assert_allclose
from craco.tracing.tracing import *

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


class FakePlan:
    def __init__(self):
        self.npix = 256
        self.nt = 256
        self.tsamp_s = 1.7*u.millisecond
        self.first_tstart = Time('2023-09-05 12:12:13')
        self.fmin = 1e9
        self.fmax = self.fmin+288e3
        self.craco_wcs = craco_wcs.CracoWCS(SkyCoord('00h00m00s -30d00m00s'),
                                            self.first_tstart,
                                            (Angle('30arcsec'),Angle('30arcsec')),
                                            self.npix,
                                            self.tsamp_s)
        self.wcs = self.craco_wcs.wcs2

@fixture
def plan():
    return FakePlan()


def write(writer, N=8192):
    d = np.ones(N, dtype=CandidateWriter.raw_dtype)
    d['snr'] *= 5000
    d['time'] = np.arange(N)
    plan = FakePlan()
    raw_noise_level = 100.237862348756234578
    iblk = 1

    icands = writer.interpret_cands(d, iblk, plan, raw_noise_level)
    writer.write_cands(icands)
    writer.close()
    return icands

def test_write_text(plan):
    writer = CandidateWriter('test.txt', plan.first_tstart, overwrite=True)
    writer.fout.flush()
    header = open('test.txt', 'r').read()
    print(header)
    assert header[0] == '#'
    write(writer)

def test_write_npy(plan):
    writer = CandidateWriter('test.npy', plan.first_tstart, overwrite=True)
    write(writer)

def test_write_gz(plan):
    writer = CandidateWriter('test.txt.gz', plan.first_tstart, overwrite=True)
    write(writer)

def test_write_text_n_0(plan):
    writer = CandidateWriter('test.txt', plan.first_tstart, overwrite=True)
    writer.fout.flush()
    header = open('test.txt', 'r').read()
    print(header)
    assert header[0] == '#'
    write(writer,0)

def test_time_conversion(plan):
    writer = CandidateWriter('test.txt', plan.first_tstart, overwrite=True)
    icands = write(writer)

    # I changed the code a bit, so I just ant to check I haven't ruined it with respect ot the old code
    # this is what the old code used to do
    old_mjd = plan.first_tstart.utc.mjd + icands['obstime_sec'].astype(writer.out_dtype['mjd']) / 3600 / 24
    new_mjd = icands['mjd']
    diff = (new_mjd - old_mjd)*86400
    print(diff)
    #assert np.all(diff == 0), f'Badd difference {diff}'
    assert_allclose(old_mjd, new_mjd)

def test_latency(plan):
    writer = CandidateWriter('test.txt', plan.first_tstart, overwrite=True)
    icands = write(writer)

    # I changed the code a bit, so I just ant to check I haven't ruined it with respect ot the old code
    # this is what the old code used to do
    assert np.all(icands['latency_ms'] >0 )

    now = plan.first_tstart
    # this is a bit insane, setting now to the begnning, because the latencie will be negative, but what the heck
    
    icands = writer.update_latency(icands, now=now) 
    #assert icands[0]['obstime_sec'] == 0.0
    #assert icands[0]['latency_ms'] == 0
    assert np.all(icands['latency_ms'] < 0)

def test_can_trace_candidates():
    cands = np.zeros((1), CandidateWriter.out_dtype)
    cand = cands[0]
    args = {k:cand[k] for k in cand.dtype.names}
    fname = 'trace_array_with_cand_instant.json'
    t = Tracefile(fname, type='array')
    e = InstantEvent('Candidate', ts=None,args=args, s='g')
    t += e
    t.close()
    


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
