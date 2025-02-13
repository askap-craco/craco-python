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
from craco.candidate_writer import CandidateWriter
import pandas as pd

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

dtype = CandidateWriter.out_dtype

search_cas_fil_dtype = np.dtype([
    ('SNR','<f4'),
    ('boxcar','<u1'),
    ('DM','<f4'),
    ('samp','<f4'),
    ('ngroup','<f4'),
    ('ncluster','<f4'),
    ('boxcar_ms','<f4'),
    ('DM_pccc','<f4'),
    ('time_s','<f4'),
    ('mjd_inf','<f4'),
    ('mjd_lower_edge','<f4')
])

out_dtype_list = [
            ('snr', '<f4'),
            ('lpix', '<u1'),
            ('mpix', '<u1'),
            ('boxc_width', '<u1'),
            ('time', '<u1'),
            ('dm', '<u2'),
            ('iblk', '<u4'),            #Saturates after 12725 days
            ('rawsn', '<i2'),
            ('total_sample', '<u4'),    #Saturates after 50 days
            ('obstime_sec', '<f4'),     #Saturates after 25 days
            ('mjd', '<f8'),
            ('dm_pccm3', '<f4'),
            ('ra_deg', '<f4'),
            ('dec_deg', 'f4'),
            ('ibeam', '<u1'), # beam number
            ('latency_ms', '<f4') # latency in milliseconds. Can be update occasionally
        ]

all_dtypes = [CandidateWriter.out_dtype,
              CandidateWriter.out_dtype_short,
              search_cas_fil_dtype]

def load_cands(fname, maxcount=None, fmt='numpy'):
    '''
    Load candidates from file
    if fmt=='numpy' (default) returns numpy structured array.
    if fmt=='pandas' returns pandas dataframe
    '''
    c = None
    for dt in all_dtypes:
        try:               
            c = np.loadtxt(fname, dtype=dt, max_rows=maxcount)
            break
        except ValueError: # usually happens if the input file is missing the last 2 columns
            pass # that dtype didn't work

    if c is None:
        raise ValueError(f'Could not load file{fname} Unkonwn dtype')
        
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
    
    s =  f"snr={c['snr']:0.1f} width={c['boxc_width']} dm={c['dm']}={c['dm_pccm3']:0.1f}pc/cm3 lm={c['lpix']},{c['mpix']}={coords} iblk={c['iblk']} time={c['time']} obssec={c['obstime_sec']:0.4f} total_samp={c['total_sample']} mjd={t.mjd}={t.utc.isot} beam={c['ibeam']}"
    
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
    parser.add_argument('--ncandvblk', action='store_true', help='Also plot ncand v block')
    parser.add_argument('-s','--sn-gain', help='Scale marker size S/N by this factor', type=float, default=1.0)
    parser.add_argument('--newfig', action='store_true', help='Start new figure for each file')
    parser.add_argument('--raw-units', action='store_true', help='Use raw units')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    fig, ax = pylab.subplots(2,2)
    tolerance = 3

    if values.raw_units:
        snfields = ('rawsn', 'rawsn')
        dmfields = ('dm','idm')
        tfields = ('total_sample', 'total_sample')
    else:
        snfields = ('snr', 'S/N')
        dmfields = ('dm_pccm3', 'DM (pc/cm3)')
        tfields = ('obstime_sec', 'Obstime (sec)')

    for ifile, f in enumerate(values.files):
        if ifile >= 1 and values.newfig:
            fig, ax = pylab.subplots(2,2)
            all_artists[:] = []
            
        fig.suptitle(f)
        c = load_cands(f)
        print(f'Loaded {len(c)} candidates from {f}')
        
        if values.threshold is not None:
            c = c[c['snr'] >= values.threshold]

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

        snhist.hist(c[snfields[0]], histtype='step', bins=50)
        snhist.set_xlabel(snfields[1])
        snhist.set_ylabel('count')

        dmhist.hist(c[dmfields[0]], histtype='step', bins=50)
        dmhist.set_xlabel(dmfields[1])
        dmhist.set_ylabel('count')

        ms = c[snfields[0]]**2 * values.sn_gain

        points1 = candvt.scatter(c[tfields[0]], c[dmfields[0]]+1, s=ms, picker=tolerance)
        all_artists.append(CandfileArtist(candfile, points1))
        candvt.set_yscale('log')
        candvt.set_xlabel(tfields[1])
        candvt.set_ylabel(f'1+{dmfields[1]}')

        if values.pixel is None:
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
        else:
            line = candimg.plot(c[tfields[0]], c[snfields[0]], picker=tolerance, ls='-', marker='o')
            all_artists.append(CandfileArtist(candfile, line))
            candimg.set_xlabel(tfields[0])
            candimg.set_ylabel(snfields[1])



        fig.tight_layout()

        if values.ncandvblk:
            f2, axs = pylab.subplots(1,1)
            f2.suptitle(f)
            imax = c['iblk'].max()
            ncand_vs_blk = [sum(c['iblk'] == iblk) for iblk in range(imax)]
            axs.plot(ncand_vs_blk)
            axs.set_xlabel('iblk')
            axs.set_ylabel('ncand')




    candvt.set_ylim(1, None)
    fig.canvas.callbacks.connect('pick_event', on_pick)
    print('Showing')
    pylab.show()
    if values.output:
        pylab.savefig(values.output)
        
    

if __name__ == '__main__':
    _main()
