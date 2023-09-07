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


def write(writer, N=8192):
    d = np.ones(N, dtype=CandidateWriter.raw_dtype)
    d['snr'] *= 5000
    plan = FakePlan()
    raw_noise_level = 100.237862348756234578
    icands = writer.interpret_cands(d, 1, plan, plan.first_tstart, raw_noise_level)
    writer.write_cands(icands)
    writer.close()

def test_write_text():
    writer = CandidateWriter('test.txt', overwrite=True)
    writer.fout.flush()
    header = open('test.txt', 'r').read()
    print(header)
    assert header[0] == '#'
    write(writer)

def test_write_npy():
    writer = CandidateWriter('test.npy', overwrite=True)
    write(writer)

def test_write_gz():
    writer = CandidateWriter('test.txt.gz', overwrite=True)
    write(writer)

def test_write_text_n_0():
    writer = CandidateWriter('test.txt', overwrite=True)
    writer.fout.flush()
    header = open('test.txt', 'r').read()
    print(header)
    assert header[0] == '#'
    write(writer,0)


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
