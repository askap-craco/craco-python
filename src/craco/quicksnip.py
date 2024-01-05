#!/usr/bin/env python
"""
Snip out a bit of a uvfits file

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco import uvfits_meta, fixuvfits
from astropy.io import fits

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def snip(infile, outfile, startidx, nblk, metadata_file=None):
    hdr = fits.getheader(infile)
    inf = uvfits_meta.open(infile, metadata_file=metadata_file)
    with open(outfile, 'wb') as fout:
        fout.seek(0,0)
        hdrb = bytes(hdr.tostring(), 'utf-8')
        fout.write(hdrb)
        gcount = 0
        tbytes = len(hdrb)
        assert len(hdrb) % 2880 == 0
        log.debug('Opened %s to %s start=%s nblk=%s hdr=%s vissize=%s nsamp=%s', infile, outfile, startidx, nblk, len(hdrb), inf.vis.size, inf.nsamps)
        for iblk, dout in enumerate(inf.fast_raw_blocks(istart=startidx, nsamp=nblk, nt=1, raw_date=True)):
            dout.tofile(fout)
            gcount += dout.size
            tbytes += dout.size*dout.dtype.itemsize


        hdr['GCOUNT']  = gcount
        fout.seek(0,0)
        fout.write(bytes(hdr.tostring(), 'utf-8'))
        fout.flush()
        log.debug('Finished writing %s. Gcount=%s', outfile, gcount)
        
    fixuvfits.fix_length(outfile)
    inhdu = fits.open(infile)
    outhdu = fits.open(outfile, 'append')
    for it, table in enumerate(inhdu[1:]):
        row = table.data[0]
        if table.name == 'AIPS SN' and row['SOURCE'].strip() == 'UNKNOWN':
            row['SOURCE'] = inf.target_name
            row['RAEPO'] = inf.target_skycoord.ra.deg
            row['DECEPO'] = inf.target_skycoord.dec.deg
            log.info('Replaced UNKNOWN source with %s %s', inf.target_name, inf.target_skycoord.to_string('hmsdms'))
        outhdu.append(table)
        
    outhdu.flush()

    outhdu.close()
    inhdu.close()
    



def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs=1)
    parser.add_argument('--metadata', help='Metadata file')
    parser.add_argument('--start-samp', type=int, help='Start sample', default=0)
    parser.add_argument('-N','--nsamp', type=int, help='Number of samples', default=1)
    parser.add_argument('-O','--outfile', help='Output file', required=True)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    snip(values.files[0],
         values.outfile,
         values.start_samp,
         values.nsamp,
         values.metadata)
    

if __name__ == '__main__':
    _main()
