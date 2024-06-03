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
from craco.dataframe_streamer import DataframeStreamer
import pandas as pd
from craco.candidate_writer import CandidateWriter

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def test_dataframe_streamer():
    n = 10
    fout = '/tmp/test.csv'
    npdata = np.ones(n, dtype=CandidateWriter.out_dtype)
    streamer = DataframeStreamer(fout)
    dfdata = pd.DataFrame(npdata)
    streamer.write(dfdata)
    streamer.write(dfdata+1)
    streamer.close()
    df = pd.read_csv(fout)
    print(df)
    assert len(df) == 2*n
    for c in npdata.dtype.names:
        assert c in list(df.columns)

    

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
