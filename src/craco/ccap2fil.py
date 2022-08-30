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

        
        filename = f'{suffix}.fil'
        log.debug('Opening file %s with header %s', filename, hdr)
        f = SigprocFile(filename, 'wb', hdr)
        self.files[key] = f

    def get_file(self, key):
        return self.files[key]

    def put(self, key, data):
        f = self.get_file(key)
        log.debug(f'Writing {data.shape} {data.dtype} to {key} {f.filename} mean={data.mean()} rms={data.std()}')
        data.astype(np.float32).tofile(f.fin)

    def close():
        for key, f in self.files:
            f.fin.close()

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-a','--autos', nargs='?', const='autos', help='Write  antenna autos to separate files')
    parser.add_argument('-s','--sum-autos',nargs='?', const='ics', help='Write sum of antenna autos')
    #parser.add_argument('-c','--crossamp', nargs='?', const='ca', help='Write baseline amps to separate files (lots of files')
    parser.add_argument('-b','--sum-crossamp', nargs='?', const='cas', help='Write sum of baseline cross amplitude')
    parser.add_argument(dest='files', nargs='+')
    parser.add_argument('--polsum', action='store_true', help='Sum polarisations')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

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

    products, revproducts, auto_products, cross_products = merger.indexes
    beam = 0
        
    for fid, dout in merger.block_iter():
        (nchan,nbeam,ntime,nbl,npol,_) = dout.shape
        if values.polsum:
            dout = dout.mean(axis=4, keepdims=True)
            
        assert nbeam == 1, 'Dont support more than one beam yet'

        if values.autos:
            for iant,idx in enumerate(auto_products):
                d = dout[:,beam,:,idx,:,0].data.astype(np.float32)
                # d shape is (nchan, nt, 1)
                d = d[...,0].T
                files.put(('auto',iant), d)

        if values.sum_autos:
            d = dout[:,beam,:,auto_products,:,0]
            log.debug(f'ICS dshape={d.shape}/{d.dtype} dout.shape={dout.shape}/{dout.dtype}')
            d = d.mean(axis=0).data[...,0].T
            files.put(('ics',0),d)

        if values.sum_crossamp:
            d = dout[:,beam,:,cross_products,:,:]
            dabs = np.sqrt(d[...,0]**2 + d[...,1]**2)
            dcsum = dabs.mean(axis=0).data[...,0].T
            log.debug(f'CROSS shape {dout.shape} {d.shape} {dcsum.shape} {dabs.shape}')
            files.put(('crossum',0),dcsum)
            
            
        
    

if __name__ == '__main__':
    _main()
