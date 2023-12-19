#!/usr/bin/env python
"""
A metadata source class that takes most of its stuff from a metafile but calculates UVW 
from a calc11 file

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.metadatafile import MetadataFile
from astropy import units as u


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class CalcMetafile(MetadataFile):
    def __init__(self, fname, calc_results):
        super().__init__(fname)
        self._calc_results = calc_results

    def uvw_at_time(self, mjd, beam=None):
        assert beam is None, 'CalcMetafile only knows about 1 beam'

        offset = 90*0.110*u.second*0 # An integration
        d = self._calc_results.scans[0].eval_src0_poly((mjd + offset).utc.value)
        nant = 36
        uvw = np.zeros((nant,3))
                
        for iant, ant in enumerate(self._calc_results.telnames):
            auvw = d[ant]
            # I dont' know why calc and the metadata file disagree by a minus
            # sign but they do
            uvw[iant, 0] = -auvw['U (m)']
            uvw[iant, 1] = -auvw['V (m)']
            uvw[iant, 2] = -auvw['W (m)']

        return uvw

    def flags_at_time(self, mjd):
        return np.zeros(self.nant, dtype=bool)
        

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
