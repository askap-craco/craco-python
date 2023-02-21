#!/usr/bin/env python
"""
CRACO data directory information

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import glob
import pandas as pd

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def get_dir_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


class DataDirs:
    def __init__(self):
        self.cracodata = os.environ['CRACO_DATA'] # local data dir when recording usuall /data/fast/craco or /data/big/craco
        assert self.cracodata is not None, 'Set CRACO_DATA environment variable'
        assert os.path.isdir(self.cracodata), f'CRACO DATA dir {self.cracodata} not a directory'
        
        self.disktype = self.cracodata.split('/')[2] #
        assert self.disktype == 'fast' or self.disktype == 'big', f'Unexpected disk type in cracodata environtment variable {self.cracodata}={self.disktype}'

    def node_dir(self, sid):
        ddir = f'/data/seren-{sid:02d}/{self.disktype}/craco'
        return ddir


    @property
    def node_dirs(self):
        for s in range(1,11):
            sdir = self.node_dir(s)
            yield sdir

    @property
    def node_names(self):
        for s in range(1,11):
            yield f'seren-{s:02d}'
        
    @property
    def schedblocks_by_node(self):
        all_sbs = []
        for ddir in self.node_dirs:
            all_sbs.append(list(map(os.path.basename, glob.glob(os.path.join(ddir, 'SB*')))))

        return all_sbs
    
    @property
    def all_schedblocks(self):
        allsbs = set()
        for sbs in self.schedblocks_by_node:
            allsbs.update(sbs)

        return sorted(list(allsbs))

    def sb_sizes(self, sb):
        sizes = np.array(list(map(get_dir_size, [os.path.join(ddir, sb) for ddir in self.node_dirs])))
        return sizes

    def sb_scan_dumps(self, sb):
        '''
        Returns a list of data dumps in teh given SB
        of the form scans/00/20230204105616
        '''
        topdir = os.path.join(self.cracodata, sb)
        if not os.path.isdir(topdir):
            raise ValueError(f'SB {sb} not found in {self.cracodata}')

        globpath = os.path.join(topdir, 'scans/*/*')
        sb_dumps = glob.glob(globpath)
        log.debug('Globbing path %s had %d results', globpath, len(sb_dumps))

        
        # strip leding bits including /
        sb_dumps = [d[len(topdir)+1:] for d in sb_dumps]

        return sb_dumps

    @property
    def sb_size_table(self, sbs=None):
        columns = ['SB']
        columns.extend(list(self.node_names))
        columns.append('Total')
        all_data = []
        if sbs is None:
            sbs = self.all_schedblocks
            
        for sbid in sbs:
            row = [sbid]
            sz = self.sb_sizes(sbid)/1024/1024/1024 # conver tto GB
            row.extend(sz) 
            row.append(sz.sum())
            all_data.append(row)

        df = pd.DataFrame(all_data, columns=columns)
        #df.style.concat(df.agg(['sum']).style) # add sum to botom https://stackoverflow.com/questions/21752399/pandas-dataframe-total-row needs later version of python
        return df
        

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
