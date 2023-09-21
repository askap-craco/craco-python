#!/usr/bin/env python
"""
Plot cRACO candidates roughly

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from collections import namedtuple
from astropy.coordinates import SkyCoord
from astropy.time import Time
import pandas as pd

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

dtype = np.dtype([('SNR',np.float32),
                  ('lpix', np.uint16),
                  ('mpix', np.uint16),
                  ('boxc_width', np.uint8),
                  ('time', np.int),
                  ('dm', np.int),
                  ('iblk', np.int),
                  ('rawsn', np.int),
                  ('total_sample', np.int),
                  ('obstime_sec', np.float32),
                  ('mjd', np.float64),
                  ('dm_pccm3', np.float32),
                  ('ra_deg', np.float64),
                  ('dec_deg', np.float64)])

def load_cands(fname, maxcount=None, fmt='numpy'):
    '''
    Load candidates from file
    if fmt=='numpy' (default) returns numpy structured array.
    if fmt=='pandas' returns pandas dataframe
    '''

    c = np.loadtxt(fname, dtype=dtype, max_rows=maxcount)
    if fmt == 'numpy':
        if len(c.shape) == 0: # psycho np.loadtxt returns a length 0 array with only 1 row!
            c.shape = (1,)
            
    elif fmt == 'pandas':
        c = pd.DataFrame(c)
    else:
        raise ValueError(f'Unknown format {fmt}')
    
    return c

CandInputFile = namedtuple("CandInputFile", 'filename candidates')
CandfileArtist = namedtuple("CandfileArtist", 'candfile artist')

def cand2str(c):
    coord = SkyCoord(c['ra_deg'],c['dec_deg'], frame='icrs', unit='deg')
    coords = coord.to_string('hmsdms')
    t = Time(c['mjd'], scale='utc', format='mjd')
    
    s =  f"SNR={c['SNR']:0.1f} width={c['boxc_width']} dm={c['dm']}={c['dm_pccm3']:0.1f}pc/cm3 lm={c['lpix']},{c['mpix']}={coords} iblk={c['iblk']} time={c['time']} obssec={c['obstime_sec']:0.4f} total_samp={c['total_sample']} mjd={t.mjd}={t.utc.isot}"
    
    return s

# should make a class but maket this global for now
all_artists = []
def on_pick(event):
    # shoudl just return one thing
    candf= next(filter(lambda x: x.artist == event.artist, all_artists))
    candfile = candf.candfile
#    print('Artist picked:', event.artist)
#    print(f'{len(ind)} vertices picked:{ind}')
#    print(event.mouseevent)
#    print(candf)

    for i in event.ind:
        print(f'{candfile.filename}[{i}] {cand2str(candfile.candidates[i])}')


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-o','--output', help='Output file name')
    parser.add_argument('-t','--threshold', help='Cut candidates below this threshold',  type=float)
    parser.add_argument('-c','--maxcount', help='Maximum number of rows to load', type=int)
    parser.add_argument('-p','--pixel', help='Comma serparated pixel to look at')
    parser.add_argument('-d','--dm', help='DM to filter for', type=float)
    parser.add_argument('-s','--sn-gain', help='Scale marker size S/N by this factor', type=float, default=1.0)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fig, ax = pylab.subplots(2,2)
    tolerance = 3

    for f in values.files:
        c = load_cands(f)
        print(f'Loaded {len(c)} candidates from {f}')
        
        if values.threshold is not None:
            c = c[c['SNR'] >= values.threshold]

        if values.pixel is not None:
            lpix, mpix = map(int, values.pixel.split(','))
            c = c[(c['lpix'] == lpix) & (c['mpix']==mpix)]

        if values.dm is not None:
            c = c[c['dm'] == values.dm]
            
        if len(c) == 0:
            print(f'{f} contained no candidates after applying thresholds')
            continue
        candfile = CandInputFile(filename=f, candidates=c)
        dmhist = ax[0,0]
        snhist = ax[0,1]
        candvt = ax[1,0]
        candimg = ax[1,1]

        snhist.hist(c['SNR'], histtype='step', bins=50)
        snhist.set_xlabel('S/N')
        snhist.set_ylabel('count')

        dmhist.hist(c['dm_pccm3'], histtype='step', bins=50)
        dmhist.set_xlabel('DM (pc/cm3)')
        dmhist.set_ylabel('count')

        ms = c['SNR']**2 * values.sn_gain

        points1 = candvt.scatter(c['obstime_sec'], c['dm_pccm3']+1, s=ms, picker=tolerance)
        all_artists.append(CandfileArtist(candfile, points1))
        candvt.set_yscale('log')
        candvt.set_xlabel('Obstime (sec)')
        candvt.set_ylabel('1+DM (pc/cm3)')

        points2 = candimg.scatter(c['ra_deg'],c['dec_deg'], s=ms, picker=tolerance)
        all_artists.append(CandfileArtist(candfile, points2))
        
        dec = c['dec_deg']
        ra = c['ra_deg']
        if len(c) == 0:
            print(f, 'is empty')
        else:
            print(f, 'decrange', dec.max() - dec.min(), 'rarange', ra.max() - ra.min())

        candimg.set_xlabel('RA (deg)')
        candimg.set_ylabel('Dec (deg)')

        fig.tight_layout()

    candvt.set_ylim(1, None)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    print('Showing')
    pylab.show()
    if values.output:
        pylab.savefig(values.output)
        
    

if __name__ == '__main__':
    _main()
