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
from craft.parset import Parset
import scipy
from collections import namedtuple
from astropy import units as u

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def get_antennas(pset):
    Antenna = namedtuple('Antenna',('antname','antpos'))
    ants = []
    for antno in range(1,36+1):
        poskey = f'common.antenna.ant{antno}.location.itrf'
        pos = tuple(map(float, pset[poskey]))
        namekey = f'common.antenna.ant{antno}.name'
        name = pset[namekey]
        ant = Antenna(antname=name, antpos=pos)
        ants.append(ant)

    return ants
        
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Converts cardcap to UVFITS', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-m','--metadata', help='Path to schedblock metdata .json.gz file', required=True)
    parser.add_argument('-f', '--fcm', help='Path to FCM file for antenna positions', required=True)
    parser.add_argument('-o','--output', help='Output fits file', default='output.uvfits')
    parser.add_argument('-N','--nblocks', help='Maximum number of blocks to write', default=-1, type=int)
    parser.add_argument('-D','--dtype', help='Data type of output f4 is 32 bit float. i2 is 16 bit integer', default='f4', choices=('f4','i2', 'i4'))
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dt = np.dtype('>'+values.dtype)
    log.info(f'Dtype {values.dtype} {type(values.dtype)} dt={dt}')


    md = MetadataFile(values.metadata)
    merge = CcapMerger(values.files)
    nbeam = merge.nbeam

    antnos = merge.antnos
    beam = 0 # TODO
    source_list = list(md.sources(beam).values())
    fcm = Parset.from_file(values.fcm)
    antennas = get_antennas(fcm)
    log.debug('FCM %s contained %d antennas %s', values.fcm, len(antennas), antennas)

    # fits format is for UVW in seconds
    # we set the scale parameter in the firts format to max out the integers for
    # the longest baseline

    bmax = 6e3*u.meter
    nant = merge.nant
    inttime = merge.inttime # seconds
    tstart = merge.mjd0.value + inttime/3600/24 / 2
    time_scale = 1*u.day # I can't use inttime here, asthere's a bug in the scaling and I don't understadn the AIPS convention of 2 DATE random parameters and whether i should encode JD0
    # as midning on the first day of hte observation, or not.
    
    uvout = CorrUvFitsFile(values.output,
                           merge.fcent,
                           merge.foff,
                           merge.nchan,
                           merge.npol,
                           tstart,
                           source_list,
                           antennas,
                           instrume='CRACO',
                           output_dtype=dt,
                           bmax=bmax,
                           time_scale=time_scale)

    try:
        for iblk, (fid, blk) in enumerate(merge.block_iter()):
            log.debug('Starting block %s FID=%s shape=%s', iblk, fid, blk.shape)
            if values.nblocks == iblk:
                break
        
            weights = 1-blk.mask.astype(np.float32) # 0 if flagged. 1 if OK.
            blidx = 0
            mjd = merge.fid_to_mjd(fid)
            uvw = md.uvw_at_time(mjd.value)/scipy.constants.c # UVW in seconds 
            antflags = md.flags_at_time(mjd.value)
            sourceidx = md.source_index_at_time(mjd.value)+ 1 # FITS standard starts at 1
            sourcename = md.source_name_at_time(mjd.value)
            log.debug('ibld=%s mjd=%s source=%s id=%d', iblk, mjd.value, sourcename, sourceidx)

            for ia1 in range(nant):
                for ia2 in range(ia1, nant):
                    uvwdiff = uvw[ia1,beam,:] - uvw[ia2,beam,:] 
                    dblk = blk[:, beam,0,blidx,:,:]
                    wblk = weights[:, beam, 0, blidx, :, 0] # real and imaginary part should have same flag
                    if antflags[ia1] or antflags[ia2]:
                        wblk[:] = 0

                    t = iblk
                    t = None # Don't use the integration time for encoding timestamp - it doesn't work yet
                    uvout.put_data(uvwdiff, mjd.value, ia1, ia2, inttime, dblk, wblk, sourceidx)
                    blidx += 1
    finally:
        print('Closing output file')
        uvout.close()
            

if __name__ == '__main__':
    _main()
