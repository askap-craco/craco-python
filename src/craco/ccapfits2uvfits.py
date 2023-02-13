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
from craco.cardcapfile import NBEAM
from craco.cardcapmerger import CcapMerger
from craft.corruvfits import CorrUvFitsFile
from craco.metadatafile import MetadataFile
from craft.parset import Parset
from craco.utils import get_target_beam
import scipy
from collections import namedtuple
from astropy import units as u
import warnings

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
    parser.add_argument('-b','--beam', help='Beam to dump. set to -1 for 36 beams, -2 is for the beam name in that TARGET field, else 0-35 for selected beam. Leave it along for whatevers in the file', type=int)
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
    fcm = Parset.from_file(values.fcm)
    antennas = get_antennas(fcm)
    log.info('FCM %s contained %d antennas %s', values.fcm, len(antennas), antennas)
    log.info('Merge contains %d beams: %s. Requested: %s', merge.nbeam, merge.beams, values.beam)

    # fits format is for UVW in seconds
    # we set the scale parameter in the firts format to max out the integers for
    # the longest baseline

    bmax = 6e3*u.meter
    nant = merge.nant
    inttime = merge.inttime # seconds
    tstart = merge.mjd0.value + inttime/3600/24 / 2
    time_scale = 1*u.day # I can't use inttime here, asthere's a bug in the scaling and I don't understadn the AIPS convention of 2 DATE random parameters and whether i should encode JD0
    # as midning on the first day of hte observation, or not.

    uvout_beams = []

    if values.beam is None or values.beam == -1:
        beams = [b for b in range(1, NBEAM+1)]
    elif values.beam == -2:
        targname = merge.gethdr('TARGET')
        beams = [get_target_beam(targname)]
        values.beam = beams[0]
        log.info('Outputting beam %s for target %s', beams[0], targname)
    else:
        beams = [values.beam]

    for ibeam, beam in enumerate(beams):
        full_source_list = list(md.sources(beam).values())
        meta_source_idx = md.source_index_at_time(tstart)
        if len(full_source_list) > 1:
            warnings.warn(f'Metadata contains more than 1 source. Setting source table to the source at tstart={tstart} which was {full_source_list[meta_source_idx]}')
            
        source_list = [full_source_list[meta_source_idx]]
        assert values.output.endswith('.uvfits')
        fileout = values.output.replace('.uvfits',f'_beam{beam:02d}.uvfits')
        log.info('Opening file %s', fileout)
        uvout = CorrUvFitsFile(fileout,
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
        uvout_beams.append((beam, uvout))

    try:
        for iblk, (fid, blk) in enumerate(merge.block_iter(beam=values.beam)):
            mjd = merge.fid_to_mjd(fid)
            log.debug('Starting block %s FID=%s mjd=%s shape=%s', iblk, fid, mjd, blk.shape)
            if values.nblocks == iblk:
                break
        
            weights = 1-blk.mask.astype(np.float32) # 0 if flagged. 1 if OK.
            uvw = md.uvw_at_time(mjd.value)/scipy.constants.c # UVW in seconds 
            antflags = md.flags_at_time(mjd.value)
            sourceidx = md.source_index_at_time(mjd.value)
            sourcename = md.source_name_at_time(mjd.value)
            log.debug('ibld=%s mjd=%s source=%s id=%d shape=%s fid=%s', iblk, mjd.value, sourcename, sourceidx, blk.shape, fid)
            # FITS standard starts at 1
            if sourceidx != meta_source_idx:
                log.info('Source changed from %s to %s at iblk=%s. Quitting', source_list[0], sourcename, iblk)
                break

            out_sourceidx = 1 # FITS convention is the first value is 1
             
            # blk shape is (nchan, nbeam, nint, nbl, npol, 2)
            assert blk.shape[2] == 1, f'Cant handle nint != 1 yet {blk.shape} {blk.shape[2]}'
            
            for ibeam, (beam, uvout) in enumerate(uvout_beams):
                blidx = 0
                for ia1 in range(nant):
                    for ia2 in range(ia1, nant):
                        uvwdiff = uvw[ia1,ibeam,:] - uvw[ia2,ibeam,:] 
                        dblk = blk[:, ibeam,0,blidx,:,:]
                        wblk = weights[:, ibeam, 0, blidx, :, 0] # real and imaginary part should have same flag
                        if antflags[ia1] or antflags[ia2]:
                            wblk[:] = 0

                        t = iblk
                        t = None # Don't use the integration time for encoding timestamp - it doesn't work yet
                        uvout.put_data(uvwdiff, mjd.value, ia1, ia2, inttime, dblk, wblk, out_sourceidx)
                        blidx += 1

    except:
        log.exception('Exception doing processing')
    finally:
        log.info('Closing output files')
        for (beam, uvout) in uvout_beams:
            uvout.close()
            

if __name__ == '__main__':
    _main()
