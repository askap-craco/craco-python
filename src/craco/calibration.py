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


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def pltcomplex(d):
    fig,ax = pylab.subplots(2,2)
    ax = ax.flatten()
    ax[0].imshow(np.abs(d), aspect='auto')
    ax[1].imshow(np.angle(d), aspect='auto')
    ax[1].set_title('phase')
    ax[0].set_title('Abs')
    ax[2].plot(np.abs(d).T)
    ax[3].plot(np.degrees(np.angle(d)).T,'.')
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
        print(solnarray.shape, p.shape)
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

    Returns np complex array with shape (nant, nchan, npol)
    note: those sizes may be different than what you asked for
    '''
    if os.path.isfile(fname):
        bp = plotbp.Bandpass.load(fname)
        g = bp.bandpass[0]
        npol = g.shape[-1]
        if npol == 4: # Npol is [XX, XY, YX, YY] just pick out [XX,YY]
            log.info('Removing cross pols from solution %s', fname)
            g = g[...,[0,3]]

        log.info("loaded CALIBRATION bandpass solutions from %s", fname)

    else:
        if fname.endswith('/'): # remove traiilng slash if prsent
            fname = fname[:-1]
            
        miriadsol = MiriadGainSolutions(fname)
        miriad_bp = miriadsol.bp_real + 1j*miriadsol.bp_imag
        miriad_g = miriadsol.g_real + 1j*miriadsol.g_imag
        miriad_gbp = miriad_bp*miriad_g
        nchan = miriad_gbp.shape[0]
        # convert to (nant,nchan,npol) order
        miriad_gbp = np.transpose(miriad_gbp, [1,0,2])
        g = miriad_gbp

        log.info("loaded MIRIAD bandpass solutions from %s", fname)


    # Mask everything that is zero
    
    #g = np.ma.masked_where(g, np.abs(g) != 0)
    g = np.ma.masked_equal(g,0) # This seems to work, even though I can't get the abs to work
    return g


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
        g = load_gains(f)
        print(f'File {f} has bandpass shape {g.shape}')
        pltcomplex(g[...,0])

    pylab.show()
    

if __name__ == '__main__':
    _main()
    
