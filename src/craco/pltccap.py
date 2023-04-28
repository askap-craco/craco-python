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
from craco.cardcapfile import CardcapFile

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def plot(f, values):
    cc = CardcapFile(f)
    fid, d = next(cc.frame_iter(beam=0))
    print(f, fid, d.shape, d.dtype, d['data'].shape)
    print(cc.mainhdr.get('DSPVER','UNKNOWN'), cc.mainhdr.get('FWDIR','UNKNOWN'))
    (products, revproducts, auto_products, cross_products) = cc.indexes
    (nc, nt1) = d.shape
    nt2, nbl, npol, _   = d[0,0]['data'].shape
    ds = d['data'].reshape(nc, nt1*nt2, nbl, npol, 2)
    autos = ds[:,:,auto_products,:,0]
    cross = ds[:,:,cross_products,...]
    dmean = autos.mean(axis=2)
    dcomp = cross[...,0] + 1j*cross[...,1]

    autos_n = autos[autos != 0]
    cross_rn = dcomp.real[dcomp.real != 0]
    cross_in = dcomp.imag[dcomp.imag != 0]
    print(f'Mean/Std: Autos: {autos_n.mean()}/{autos_n.std()} Cross real: {cross_rn.mean()}/{cross_rn.std()} imag={cross_in.mean()}/{cross_in.std()}')
    

    nant = autos.shape[0]
    adcomp = abs(dcomp)
    if np.any(ds[:,:,auto_products,:,1] != 0):
        import warnings
        warnings.warn(f'File {f} contains autos with nonzero imaginary part')
    print(autos.shape, dmean.shape, cross.shape, dcomp.shape)
    fig, ax = pylab.subplots(2,3)
    for pol in range(npol):
        ax[0,0].plot(dmean[...,pol].flatten(), label=f'mean pol={pol}')
        ax[0,0].plot(autos[...,pol].std(axis=2).flatten(), label=f'std pol={pol}')
        ax[0,0].plot(autos[...,pol].min(axis=2).flatten(), label=f'min pol={pol}')
        ax[0,0].plot(autos[...,pol].max(axis=2).flatten(), label=f'max pol={pol}')

        da = autos[...,pol].flatten()
        ax[0,2].hist(da[da!=0], label=f'auto pol={pol}', histtype='step')
        
        ax[1,0].plot(abs(dcomp).mean(axis=1).flatten(), label=f'Cross mean pol={pol}')
        ax[1,0].plot(abs(dcomp).std(axis=1).flatten(), label=f'Crossamp std pol={pol}')
        dr = dcomp[...,pol].real.flatten()
        di = dcomp[...,pol].imag.flatten()

        ax[1,2].hist(dr[dr!=0],label=f'Cross real pol={pol}', histtype='step')
        ax[1,2].hist(di[di!=0],label=f'Cross imag pol={pol}',histtype='step')


    ax[0,1].imshow(autos.reshape(nant*nc, -1))
    #ax[1,1].imshow(adcomp.reshape(nc*nt1*nt2,-1))
                     

    # plot labels - it's a bit busy
    for iant in range(nant):
        x = iant*nc + nc/2
        y = dmean[iant,...].max()
        #ax[0].text(x,y, f'ak{iant:02d}', ha='center', va='bottom')
        
    ax[0,0].set_xlabel(f'ant({nant}) x chan ({nc})')
    ax[0,0].legend()
    ax[0,2].legend()
    ax[0,0].set_title('Autocorrelation - real part')

    ax[1,0].legend()
    ax[1,2].legend()

    ax[1,1].imshow(abs(dcomp[:,:,:,0].reshape(nc*nt1*nt2,-1)), aspect='auto')
    ax[1,1].set_xlabel('Baseline')
    ax[1,1].set_ylabel(f'chan({nc}) x sample({nt1*nt2})')
    

    
    fig.suptitle(f)
    pylab.show()
    
    
    

    

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


    for f in values.files:
        plot(f, values)
    

if __name__ == '__main__':
    _main()
