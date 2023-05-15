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
from craft.craco import printstats,bl2ant
from craft.craftcor import MiriadGainSolutions
from craco import plotbp
import warnings
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def pltcomplex(d, x=None):
    fig,ax = pylab.subplots(2,2)
    ax = ax.flatten()
    if x is None:
        x = np.arange(d.shape[1])
    ax[0].imshow(np.abs(d), aspect='auto')
    ax[1].imshow(np.angle(d), aspect='auto')
    ax[1].set_title('phase')
    ax[0].set_title('Abs')
    ax[2].plot(x, np.abs(d).T)
    ax[3].plot(x, np.degrees(np.angle(d)).T,'.')
    return fig,ax
    

def gains2solarray(plan, soln, npol=2):
    '''
    Retuns a complex array that can multiply with an input data cube to calbrate it
    shape: (nbl, nchan, npol, 1) - assumes the final axis (1) is time, so 
    np broadcasting rules will apply the same calibration to all times
    @param :plan: plane with baseline order
    @param :soln: gain solution from load_gains()
    '''
    solnarray = soln2array(soln, plan.baseline_order, npol)
    return solnarray

def interpolate_gains(gains, from_freqs, to_freqs):
    '''
    interpolate the gains array with gains andf requencyes to a new set of frequencies
    @param gains (nant, nf, npol) complex gains array
    @param froM_freqs length nf array of channel freqs
    @param to_freqs arbitrary lenght array of channel freqs
    '''

    nant, nf, npol = gains.shape
    assert nf == len(from_freqs)
    out_gains = gains.copy()
    # interpolate in real/imaginary space - could do amp/phase but you get in trouble with phase wrapping, maybe/
    # sometimes minor frequency labelling errors give you grief. We'll try to be a bit tolerant
    if to_freqs.min() < from_freqs.min() or to_freqs.max() > from_freqs.max():
        log.info('Requested freqs out of range: from: %s-%s to %s-%s',
                 from_freqs.min(), from_freqs.max(),
                 to_freqs.min(), to_freqs.max())

    re_interp = interp1d(from_freqs, gains.real, axis=1, fill_value='extrapolate')
    im_interp = interp1d(from_freqs, gains.imag, axis=1, fill_value='extrapolate')
    out_gains = re_interp(to_freqs) + 1j*im_interp(to_freqs)
    out_gains = np.ma.masked_where(abs(out_gains) == 0, out_gains)
    
    return out_gains
    

def soln2array(soln, baseline_order, npol = 2):
    '''
    Retuns a complex array that can multiply with an input data cube to calbrate it
    shape: (nbl, nchan, npol, 1) - assumes the final axis (1) is time, so 
    np broadcasting rules will apply the same calibration to all times
    @param :plan: plane with baseline order
    @param :soln: gain solution from load_gains()
    '''
    nbl = len(baseline_order)
    nant, nf, npol = soln.shape
    solnarray = np.zeros((nbl, nf, npol), np.complex64)
    mask = np.zeros((nbl, nf, npol), dtype=bool)
    for ibl, blid in enumerate(baseline_order):
        a1,a2 = bl2ant(blid)
        s1 = soln[a1-1,:,:]
        s2 = soln[a2-1,:,:]
        p = s1*np.conj(s2)
        #print(solnarray.shape, p.shape)
        solnarray[ibl,...] = p[:]
        mask[ibl,...] = p.mask[:]

    solnarray[solnarray != 0] = 1/solnarray[solnarray != 0]

    # update shape
    solnarray.shape = (nbl, nf, npol, 1)
    mask.shape = solnarray.shape
    solnarray = np.ma.masked_array(solnarray, mask, fill_value=0+0j)

    return solnarray

def load_gains(fname):
    '''
    Loads calibration data
    If fname points to a file, it tries to load as a "calibration" file from plotbp
    Else, it tries to load a set of miraid export gain files given 'fname' as the root name

    Returns (np complex array with shape (nant, nchan, npol), np.array(nchan) of frequncies in Hz)
    note: those sizes may be different than what you asked for
    But nant = 36 always, so we know that the non-existent natennas should be flagged
    '''
    if os.path.isfile(fname) and fname.endswith('.bin'):
        bp = plotbp.Bandpass.load(fname)
        g = bp.bandpass[0]
        npol = g.shape[-1]
        if npol == 4: # Npol is [XX, XY, YX, YY] just pick out [XX,YY]
            log.info('Removing cross pols from solution %s', fname)
            g = g[...,[0,3]]

        log.info("loaded CALIBRATION bandpass solutions from %s", fname)

        # Values are nan if the antenna is missing. We'll replace with zeros here and
        # the maske array part at the bottom of tthis function will mask them out
        g[np.isnan(g)] = 0

        freqfile = fname.replace(".bin", ".freq.npy")

        if os.path.exist(freqfile):
            freqs = np.load(freqfile)

        else:
            # get solutions from casa
            msfile = fname.replace('.bin','.ms')
            freqs = None

            # load channel frequencies from associated casa ms spectral
            if not os.path.isdir(msfile):
                raise ValueError(f'MS file missing to load frequencies {msfile}')

            log.info('Loading frequencies from %s', msfile)
            import casatools
            tb = casatools.table()
            tb.open(os.path.join(msfile, 'SPECTRAL_WINDOW'))
            assert tb.nrows() == 1, f'Expected only 1 spectral window. got {tb.nrows()}'
            freqs = tb.getcol('CHAN_FREQ')[:,0] # list of all frequencies in Hz
            chan_width = tb.getcol('CHAN_WIDTH')
            assert np.all(chan_width == chan_width[0]), f'Channel widths are not all teh same {chan_width} {fname}'
            tb.close()

    elif os.path.isdir(fname): #probably miriad 
        if fname.endswith('/'): # remove traiilng slash if prsent
            fname = fname[:-1]
            
        miriadsol = MiriadGainSolutions(fname)
        miriad_bp = miriadsol.bp_real + 1j*miriadsol.bp_imag
        miriad_g = miriadsol.g_real + 1j*miriadsol.g_imag
        ntimes = miriad_g.shape[0]
        if ntimes != 1:
            warnings.warn(f'Miraid gain solutions for {fname} contain {ntimes} solutions. Just taking average')
            miriad_g = miriad_g.mean(axis=0, keepdims=True)
            
        miriad_gbp = miriad_bp*miriad_g
        nchan = miriad_gbp.shape[0]
        # convert to (nant,nchan,npol) order
        miriad_gbp = np.transpose(miriad_gbp, [1,0,2])
        g = miriad_gbp

        log.info("loaded MIRIAD bandpass solutions from %s", fname)
        freqs = miriadsol.freqs*1e9
    else:
        raise ValueError(f'Unknown calibration file type {fname}')
            

    # Mask everything that is zero
    
    #g = np.ma.masked_where(g, np.abs(g) != 0)
    g = np.ma.masked_equal(g,0) # This seems to work, even though I can't get the abs to work
    g.set_fill_value(0+0j)
    
    
    assert g.shape[0] == 36, f'First dimension of gains shoudl be nant=36. Shape was (nant, chan, npol)={g.shape}'
    
    return (g, freqs)


class CalibrationSolution:
    def __init__(self, plan):
        self.plan = plan
        values = plan.values
        self.values = values

        if values.calibration:
            gains, freqs = load_gains(values.calibration)
            log.info('Loaded calibration gains %s from calfile %s', gains.shape, plan.values.calibration)
        else: # make dummy gains
            shape = (self.plan.maxant, self.plan.nf, 2)
            mask = np.zeros(shape, dtype=bool)
            gains = np.ma.masked_array(np.ones(shape, dtype=np.complex64), mask=mask)
            freqs = plan.freqs


        assert freqs[0] == freqs.min()
        fch1 = freqs[0]
        foff = freqs[1] - freqs[0]
        nant, nchan, npol = gains.shape
        assert len(freqs) == nchan
        chan_bw_ratio = int(np.round(self.plan.foff / foff))
        
        log.info('Calibration file: fch1=%s, foff=%s nchan=%d plan: fch1=%s foff=%s nchan=%d chan_bw_ratio=%d gains.shape=%s',
                 fch1, foff, nchan,
                 plan.fmin, plan.foff, plan.nf,
                 chan_bw_ratio,
                 gains.shape)
        assert foff >0, 'Assume freq always increasing, it just makes me feel better'
        assert self.plan.foff > 0, 'Assume freq always incerasing'
        
        # check if there's a x6 divisor, which might be common
        # If yes, then average by 6
        factor = 6
        if chan_bw_ratio == factor and nchan % factor == 0:
            log.info('Averaging solution array by %s to match plan', factor)
            gains = gains.reshape(nant, nchan // factor, factor, npol).mean(axis=2)
            freqs = freqs.reshape(-1,factor).mean(axis=1)

        # to handle cases where the calibration contains more total bandwidth than the plan/data, we'll interpolate over the
        # plan frequencies
        gains = interpolate_gains(gains, freqs, self.plan.freqs)
        self.gains = gains
        
        self.__calc_solarray()

    def __calc_solarray(self):
        (nant, nchan, npol) = self.gains.shape
        self.__solarray = gains2solarray(self.plan, self.gains)
        (nbl, nchan, npol, _) = self.__solarray.shape
        assert self.plan.nf == nchan
        ntotal = self.plan.nf * self.plan.nbl
        pcflag = (ntotal - self.num_input_cells) / ntotal*100.
        snincrease = np.sqrt(self.num_input_cells)
        log.info('Loaded calibration with %s/%s valid input cells=%0.1f%% flagged. S/N increase is %s',
                      self.num_input_cells, ntotal, pcflag, snincrease)

        return self.__solarray

    @property
    def solarray(self):
        '''Returns solution array'''
        return self.__solarray

    @property
    def num_input_cells(self):
        '''
        Returns the total number of cells added together in the grid
        if we have a calibration array, it's the number of good values in the mask
        Otheerwise, it's the product of nbl, and nf from the plan
        Note: solution array may be dual polarisation, in which case this returns the single polarisation size

        '''

        if self.solarray is None:
            nsum = self.plan.nbl * self.plan.nf 
        else:
            assert 1 <= self.solarray.shape[2] <= 2
            single_sol_array = self.solarray[:,:,0,:] # single pol solution array
            nsum = single_sol_array.size - single_sol_array.mask.sum() # number of unmasked values
            assert nsum > 0, f'Invalid number of input cells: {nsum}={single_sol_array.siz}e - {sum(single_sol_array.mask)}'

        return nsum
    
    def set_channel_flags(self, chanrange, flagval: bool):
        '''
        Set the channel flags to the given flagval.
        chanrange: list of channel numbers that can be used in numpy index
        flagval: boolean. True = flagged
        updates solarray
        '''

        #g = self.gains.copy() # todo: is this necessary?
        
        # OMFG - if you do g[:,chanrange,:].mask if chanrange if a list of indexes, it doesn't work, but if it's a slice it does work. But if you do g.mask[:,chanrange,:] it works for both
        self.gains.mask[:,chanrange,:] = flagval
        return self.__calc_solarray()

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='plot calibration solutions', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    for f in values.files:
        g,freqs = load_gains(f)
        if freqs is None:
            f1 = 0
            foff = 0
        else:
            f1 = freqs[0]
            foff = freqs[1] - freqs[0]
        print(f'File {f} has bandpass shape {g.shape} f1={f1/1e6:0.3f} MHz, foff={foff/1e3:0.3f}kHz')
        pltcomplex(g[...,0])

    pylab.show()
    

if __name__ == '__main__':
    _main()
    
