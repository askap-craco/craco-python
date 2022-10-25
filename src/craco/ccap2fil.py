#!/usr/bin/env python
"""
Merge ccap files and produce filterbanks

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco.cardcap import CardcapFile
from craco.cardcapmerger import CcapMerger
from craft.sigproc import SigprocFile
from craco.cmdline import strrange

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class FileManager:
    def __init__(self, merger, values):
        self.merger = merger
        self.values = values
        self.files = {}

    def add_file(self, key, suffix):
        npolin = self.merger.npol
        if npolin == 2:
            if self.values.polsum:
                npolout = 1
            else:
                npolout = 2
        else:
            npolout = 1

        self.npolout = npolout

        for beam in self.merger.beams:
            hdr = {'nbits':32,
                   'nchans':self.merger.nchan,
                   'nifs':npolout,
                   'tsamp':self.merger.inttime,
                   'fch1':self.merger.fch1,
                   'foff':self.merger.foff,
                   'src_raj':0,
                   'src_dej':0,
                   'tstart':float(self.merger.mjd0.mjd)
            }
            
            
            filename = f'{self.values.prefix}{suffix}_b{beam:02d}.fil'
            log.debug('Opening file %s with header %s', filename, hdr)
            f = SigprocFile(filename, 'wb', hdr)
            self.files[(key, beam)] = f

    def get_file(self, key, beam=0):
        return self.files[(key, beam)]

    def put(self, key, data):
        log.debug(f'Writing {data.shape} {data.dtype} to {key} mean={data.mean()} rms={data.std()}')

        # Check last 2 dimensions of input are (NIF, NCHAN)
        assert data.shape[-2:] == (self.npolout, self.merger.nchan), f'Input shape needs to be in (T, IF, NCHAN) order. Shape was: {data.shape} expected (X, {self.npolout}, {self.merger.nchan})'

        assert data.shape[0] == len(self.merger.beams), f'Expected first axis of shape={data.shape} to have nbeams={len(self.merger.beams)} size'

        for ibeam, beam in enumerate(self.merger.beams):
            f = self.get_file(key, beam)
            data[ibeam, ...].astype(np.float32).data.tofile(f.fin)

    def close():
        for key, f in self.files:
            f.fin.close()

def antidx2num(a1, a2):
    if a1 < a2:
        ma1, ma2 = a1, a2
    else:
        ma1, ma2 = a2, a1
        

    return (ma1,ma2)


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-a','--autos', help='Write  antenna autos to separate files with this postfix - if not specified it wont write autos')
    parser.add_argument('-s','--sum-autos',default='ics', help='Write sum of antenna autos with this postfix')
    parser.add_argument('-c','--crossamp', help='Write baseline amplitudes for baselines to these 1-based antenna numbers (strrange)', type=strrange)
    parser.add_argument('-b','--sum-crossamp', default='cas', help='Write sum of baseline cross amplitude with this postfix')

    parser.add_argument('-p','--prefix', default='', help='Add this prefix before the output filename')
    parser.add_argument(dest='files', nargs='+')
    parser.add_argument('--polsum', action='store_true', help='Sum polarisations')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log.debug("Got %d files", len(values.files))
    merger = CcapMerger(values.files)
    files = FileManager(merger, values)
    nant = merger.nant

    log.debug(values)
    
    if values.autos:
        for ia in range(nant):
            files.add_file(('auto',ia), f'{values.autos}_ak{ia+1:02d}')

    if values.sum_autos:
        files.add_file(('ics',0), values.sum_autos)

    if values.sum_crossamp:
        files.add_file(('crossum',0), values.sum_crossamp)

    if values.crossamp:
        for a1 in values.crossamp: # 1 based
            for a2 in range(1,nant+1):
                if a1 == a2: # autos aren't included
                    continue

                ma1, ma2 = antidx2num(a1,a2)
                    
                files.add_file(('crossamp',(ma1,ma2)), f'cross_ak{ma1:02d}_ak{ma2:02d}')
        

    products, revproducts, auto_products, cross_products = merger.indexes
        
    for fid, dout in merger.block_iter():
        (nchan,nbeam,ntime,nbl,npol,_) = dout.shape
        log.debug('Got fid=%s shape=%s', fid, dout.shape)

        # order for sigproc filterbanks is [Time, IF, channel]
        # so we'll put the polarisation axis in a mostly correct order
        # I.e.
        # (nbeam, nbl, ntime, npol, nchan, 2)
        dout = np.transpose(dout, [1,3,2,4,0,5]).astype(np.float32)
        assert dout.shape == (nbeam, nbl, ntime, npol, nchan, 2)
        
        if values.polsum:
            dout = dout.mean(axis=3, keepdims=True)
            
        #assert nbeam == 1, 'Dont support more than one beam yet'

        if values.autos:
            for iant,idx in enumerate(auto_products):
                d = dout[:,idx,:,:,:,0]
                files.put(('auto',iant), d)

        if values.sum_autos:
            d = dout[:,auto_products,:,:,:,0]
            log.debug(f'ICS dshape={d.shape}/{d.dtype} dout.shape={dout.shape}/{dout.dtype}')
            assert d.shape[0] == len(auto_products), f'Unexpectged output shape {d.shape}'
            # OMG - specifying index array makes it the first axis
            d = d.mean(axis=0)
            files.put(('ics',0), d)

        if values.sum_crossamp:
            d = dout[:,cross_products,...]
            dabs = np.sqrt(d[...,0]**2 + d[...,1]**2)
            dcsum = dabs.mean(axis=0)
            log.debug(f'CROSS shape {dout.shape} {d.shape} {dcsum.shape} {dabs.shape}')
            files.put(('crossum',0),dcsum)

        if values.crossamp:
            for a1 in values.crossamp:
                for a2 in range(1, nant+1):
                    # warning, this is a bit off the cuff
                    if a1 == a2: # autos aren't included
                        continue
                    ma1,ma2 = antidx2num(a1,a2)
                    blid = revproducts[(ma1,ma2)]
                    d = dout[beam, blid, ...]
                    dabs = np.sqrt(d[...,0]**2 + d[...,1]**2)
                    files.put(('crossamp', (ma1, ma2)), dabs)
                    

            
            
        
    

if __name__ == '__main__':
    _main()
