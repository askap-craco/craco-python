#!/usr/bin/env python
"""
Visibility flagging class

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from iqrm import iqrm_mask

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def my_iqrm_mask(x, radius, threshold):
    '''
    Compute iqrm mask if radius is valid > 0
    '''
    
    if radius > 0:
        mask, _ = iqrm_mask(x, radius=radius, threshold=threshold)
    else:
        mask = np.zeros(len(x), dtype=bool)

    return mask

def calc_mask(ics, factor, fradius, tradius, threshold):

    # if the data have packet loss they'll be zero
    # we mask them out so we don't bais the measurement of std
    
    ics = np.ma.masked_equal(ics, 0)
    if factor != 1:
        nf, nt = ics.shape
        ics = ics.reshape(-1, factor, nt).mean(axis=1)

    tstd = ics.std(axis=0)
    fstd = ics.std(axis=1)
    fmask = my_iqrm_mask(fstd, radius=fradius, threshold=threshold)
    tmask = my_iqrm_mask(tstd, radius=tradius, threshold=threshold)
    
    return fmask, tmask

class VisFlagger:
    '''
    Configurable block-based visibility flagger

    Uses IQRM and CAS/ICS to compute masks on standard deviation across time
    
    '''

    def __init__(self, fradius, tradius, threshold, tblk=None):
        self.fradius = fradius
        self.tradius = tradius
        self.threshold = threshold
        self.tblk = tblk

    def flag_block(self, input_flat, cas, ics):

        (nbl, nf, nt) = input_flat.shape
        assert cas.shape == ics.shape
        nfcas, nt2 = cas.shape
        assert nt == nt2

        assert nfcas >= nf
        factor = nfcas // nf

        ics_fmask, ics_tmask = calc_mask(ics, factor, self.fradius, self.tradius, self.threshold)
        cas_fmask, cas_tmask = calc_mask(cas, factor, self.fradius, self.tradius, self.threshold)

        fmask = ics_fmask | cas_fmask
        tmask = ics_tmask | cas_tmask

        input_flat.mask |= fmask[np.newaxis,:,np.newaxis] | tmask[np.newaxis,np.newaxis,:]

        return input_flat
        
        
    def __call__(self, input_flat, cas, ics):
        '''
        Updates input flat mask shape (nbl, nf, nt)
        Expects cas, ics as (nf, nt) and ors the mask together
        Computes in blocks of self.tblk to capture shorter RFI
        '''

        if cas is None or ics is None:
            if self.fradius > 0 or self.tradius > 0:
                raise ValueError('Requested flagging but cas or ICS not supplied')
            
            return input_flat

        nbl, nf, nt = input_flat.shape
        tblk = self.tblk if self.tblk is not None else nt

            
        assert nt % tblk == 0, f'Invalid tblk={tblk} or nt ={nt}'
        nblk = nt //tblk

        for iblk in range(nblk):
            start = iblk*tblk
            end = start + tblk
            idx = slice(start, end)
            self.flag_block(input_flat[:,:,idx], cas[:,idx], ics[:,idx])

        return input_flat
        
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
