#!/usr/bin/env python
"""
Streams dataframes to disk. Useful for candpipe

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
import pandas as pd
import atexit

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class DataframeStreamer:
    def __init__(self, foutname):
        self.fout = open(foutname, 'w')
        self.nwrite = 0
        atexit.register(self.close)

    def write(self, df:pd.DataFrame, **kwargs):
        '''
        Write given dataframe if not none or zero length
        includes header for first non-emptty dataframe
        pass kwargs to Dataframe.to_csv

        '''

        if df is None or len(df) == 0:
            return
        
        write_header = self.nwrite == 0
        df.to_csv(self.fout, header=write_header, **kwargs)
        self.fout.flush()
        self.nwrite += len(df)

    def close(self):
        if self.fout is not None:            
            self.fout = None



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
