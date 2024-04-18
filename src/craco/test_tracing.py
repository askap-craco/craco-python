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
from craco.tracing import *
import json

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
def jsonfile(f):
    with open(f, 'rt') as fin:
        d = json.load(fin)

    return d

def check_event_list(l1, l2):
    assert len(l1) == len(l2)
    for d1, d2 in zip(l1,l2):
        for k,v2 in d2.items():
            v1 = getattr(d1,k)
            assert v1 == v2

def test_tracefile_array_is_json():
    fname = 'trace_array.json'
    t = Tracefile(fname, type='array')
    e = Event('hello', 1)
    t += e
    t.close()
    d = jsonfile(fname)
    check_event_list([e], d)

def test_tracefile_object_is_json():
    fname = 'trace_object.json'
    t = Tracefile(fname, type='object')
    e = Event('hello', 1)
    t += e
    t.close()
    d = jsonfile(fname)
    assert d['stackFrames'] == []
    check_event_list([e],d['traceEvents'])
    

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
