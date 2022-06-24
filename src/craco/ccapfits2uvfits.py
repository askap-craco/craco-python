#!/usr/bin/env python
"""
Merges cardcap fits files with metdata and writes uvfits format

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.cardcapmerger import CcapMerger
from craft.corruvfits import CorrUvFitsFile
from craco.metadatafile import MetadataFile
import scipy

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Converts cardcap to UVFITS', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-m','--metadata', help='Path to schedblock metdata .json.gz file')
    parser.add_argument('-o','--output', help='Output fits file')
    parser.add_argument('-N','--nblocks', help='Maximum number of blocks to write', default=-1, type=int)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    md = MetadataFile(values.metadata)
    merge = CcapMerger(values.files)
    antennas = [] # TODO

    beam = 0 # TODO
    source_list = list(md.sources(beam).values())

    uvout = CorrUvFitsFile(values.output,
                           merge.fcent,
                           merge.foff,
                           merge.nchan,
                           merge.npol,
                           merge.mjd0.value,
                           source_list,
                           antennas,
                           instrume='CRACO')

    nant = merge.nant
    inttime = merge.inttime
    source = 1 # TODO
    print(md.sources(beam))

    try:
        for iblk, (fid, blk) in enumerate(merge.block_iter()):
            log.debug('Starting block %s FID=%s shape=%s', iblk, fid, blk.shape)
            if values.nblocks == iblk:
                break
        
            weights = 1-blk.mask.astype(np.float32) # 0 if flagged. 1 if OK.
            blidx = 0
            mjd = merge.fid_to_mjd(fid)
            uvw = md.uvw_at_time(mjd.value)/scipy.constants.c
            for ia1 in range(nant):
                for ia2 in range(ia1, nant):

                    uvwdiff = uvw[ia1,beam,:] - uvw[ia2,beam,:] 
                    
                    dblk = blk[:, beam,0,blidx,:,:]
                    wblk = weights[:, beam, 0, blidx, :, 0] # real and imaginary part should have same flag
                    uvout.put_data(uvwdiff, mjd.value, ia1, ia2, inttime, dblk, wblk, source)
                    blidx += 1
    finally:
        uvout.close()
            

if __name__ == '__main__':
    _main()
