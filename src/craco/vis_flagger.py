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
    
    if radius > 0 and threshold > 0:
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

    def __init__(self, fradius, tradius, cas_threshold, ics_threshold, tblk=None):
        self.fradius = fradius
        self.tradius = tradius
        self.cas_threshold = cas_threshold
        self.ics_threshold = ics_threshold
        self.tblk = tblk
        self.total_tflag = 0
        self.total_fflag = 0
        self.total_tfflag = 0
        self.total_blocks = 0

    def flag_block(self, input_flat, cas, ics, use_local_cas=True):

        (nbl, nf, nt) = input_flat.shape
        assert cas.shape == ics.shape
        nfcas, nt2 = cas.shape
        assert nt == nt2, f"input_flat ({nt}) and cas ({nt2}) don't have the same nt"

        assert nfcas >= nf
        factor = nfcas // nf


        ics_fmask, ics_tmask = calc_mask(ics, factor, self.fradius, self.tradius, self.ics_threshold)

        # The input CAS is rescaled per channel, which makes it not so great at detecting RFI
        # Vivek claims it's much better to compute unrescaled CAS and use that
        # The easiest thing to do is do it here. Note; this isn't calibrated yet, so it's
        # possibly a bit biased towards antennas with higher gains, but it appears not to hurt
        # for the purposes of RFI mitigation

        if use_local_cas:
            cas = abs(input_flat).mean(axis=0)
            factor = 1
            
        cas_fmask, cas_tmask = calc_mask(cas, factor, self.fradius, self.tradius, self.cas_threshold)

        fmask = ics_fmask | cas_fmask
        tmask = ics_tmask | cas_tmask

        self.total_tflag += sum(tmask)
        self.total_fflag += sum(fmask)

        tfmask = fmask[:, np.newaxis] | tmask[np.newaxis, :]
        self.total_tfflag = tfmask.sum()

        input_flat.mask |= tfmask[np.newaxis, :, :]

        return input_flat, tfmask
        
        
    def __call__(self, input_flat, cas, ics, mask_fil_writer = None):
        '''
        Updates input flat mask shape (nbl, nf, nt)
        Expects cas, ics as (nf, nt) and ors the mask together
        Computes in blocks of self.tblk to capture shorter RFI
        Writes the computed tfmask to a craft.sigproc.SigprocFile obj
        '''

        if cas is None or ics is None:
            if self.fradius > 0 or self.tradius > 0:
                raise ValueError('Requested flagging but cas or ICS not supplied')
            
            return input_flat

        nbl, nf, nt = input_flat.shape
        tblk = self.tblk if self.tblk is not None else nt

            
        assert nt % tblk == 0, f'Invalid tblk={tblk} or nt ={nt}'
        nblk = nt //tblk

        tflag0, fflag0, tfflag0  = self.total_tflag, self.total_fflag, self.total_tfflag

        for iblk in range(nblk):
            start = iblk*tblk
            end = start + tblk
            idx = slice(start, end)
            input_flat, tfmask = self.flag_block(input_flat[:,:,idx], cas[:,idx], ics[:,idx])
            if mask_fil_writer is not None:
                np.packbits(tfmask.T.ravel()).tofile(mask_fil_writer.fin)

        tflag1, fflag1, tfflag1 = self.total_tflag, self.total_fflag, self.total_tfflag
        tflagd = tflag1 - tflag0
        fflagd = fflag1 - fflag0
        tfflagd = tfflag1 - tfflag0

        tflagpc = tflagd/nt*100
        fflagpc = fflagd/nf*100
        self.total_blocks += 1
        cum_tflagpc = self.total_tflag / (nt * self.total_blocks) *100
        cum_fflagpc = self.total_fflag / (nf * self.total_blocks) *100


        log.info('Flagging block %d T Flagged %d/%d=%0.1f  F Flagged %d/%d=%0.1f TF Flag %d/%d=%0.1f. Cumulative T flag: %0.1f Cumulative F flag=%0.1f',
                 self.total_blocks-1,
                 tflagd, nt, tflagpc, fflagd, nf, fflagpc, tfflagd, nt*nf, tfflagd/(nt*nf)*100, cum_tflagpc, cum_fflagpc)


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
