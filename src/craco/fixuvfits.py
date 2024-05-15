#!/usr/bin/env python
"""
Fix UV fits files that might be junk because not closed properly

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
from astropy.io import fits

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

FITS_BLOCK_SIZE = 2880

def fix_length(fname):
    '''
    Appends zeros to the end of a fits file to make it a multiple of FITS_BLOCK_SIZE bytes
    '''
    filesize = os.path.getsize(fname)
    with open(fname, 'ab') as fout:
        n_extra_bytes = FITS_BLOCK_SIZE - filesize % FITS_BLOCK_SIZE
        if n_extra_bytes == FITS_BLOCK_SIZE:
            n_extra_bytes = 0

        print(f'Current position {fout.tell()} writing {n_extra_bytes}')
        fout.write(bytes(n_extra_bytes))

    newsize = os.path.getsize(fname)
    print(f'Wrote {n_extra_bytes} to {fname} to make it from {filesize} {newsize}')

extra_tables = ['fq_table','an_table','su_table']

def find_extra_table_bytes(fname, lookback_blocks=16):
    '''
    Calculates how many addditional bytes at the end of the given filename
    have BINTABLEs in them
    returns 0 if there are no blocks
    '''
    pattern="XTENSION= 'BINTABLE'".encode('utf-8')
    sz = os.path.getsize(fname)
    nbytes = lookback_blocks*FITS_BLOCK_SIZE
    with open(fname, 'rb') as fin:
        fin.seek(sz - nbytes)
        thebytes = fin.read(nbytes)
        try:
            idx = thebytes.index(pattern)
            table_bytes = nbytes - idx
        except ValueError: # pattern not found
            table_bytes = 0

    return table_bytes
        

def fix_gcount(fname, groupsize=None):
    '''
    Calculate the correct GCOUNT value
    set the header to fhis value if not already done so
    The group size is the nuber of bytes per group.
    It's probably calculatable from the header somehow (pyfits des it) but
    our uvfitswriter writes a .groupsize file too, so we pick that up if gropusize is None
    
    '''
    if groupsize is None:
        with open(fname+'.groupsize', 'r') as fin:
            groupsize = int(fin.read())
        
    filesize = os.path.getsize(fname)
    hdr = fits.getheader(fname)
    #hdr = fits.header.Header.fromtextfile(fname+'.header')
    gcount = hdr['GCOUNT']
    isfixed = hdr.get('FIXED', False)
    #extra_tab_fnames = [f'{fname}.{tabname}' for tabname in extra_tables]
    #extra_tab_sizes = map(os.path.getsize, extra_tab_fnames)

    # for some reason the tables on disk have 1 extra block added
    #extra_tab_bytes = sum(extra_tab_sizes) - len(extra_tables)*FITS_BLOCK_SIZE
    extra_tab_bytes = find_extra_table_bytes(fname)
    datasize = (filesize - len(hdr.tostring()) - extra_tab_bytes)
    expected_gcount = datasize // groupsize
    print(f'File {fname}  header={len(hdr.tostring())} extra table bytes={extra_tab_bytes} filesize={filesize} datasize={datasize} GCOUNT={gcount} expected={expected_gcount} isfixed={isfixed} ')
    
    if gcount == expected_gcount:
        print(f'File {fname} has correct GCOUNT {gcount}')
    else:
        hdr['GCOUNT'] = expected_gcount
        hdr['FIXED'] = True

        with open(fname, 'r+b') as fout: # can't be 'a' as it only appends, irrepsective of seek position
            fout.seek(0,0)
            fout.write(bytes(hdr.tostring(), 'utf-8'))
            assert fout.tell() % FITS_BLOCK_SIZE == 0
            fout.flush()

def fix_tables(fname):
    # only add tables if they're missing
    extra_tab_bytes = find_extra_table_bytes(fname)
    is_missing_tables = extra_tab_bytes == 0
    
    if os.path.exists(fname+'.fq_table') and is_missing_tables:
        print('Appending tables')
        hdu = fits.open(fname, 'append')
        fq_table = fits.open(fname+'.fq_table')[1]
        an_table = fits.open(fname+'.an_table')[1]
        su_table = fits.open(fname+'.su_table')[1]
        hdu.append(fq_table)
        hdu.append(an_table)
        hdu.append(su_table)
        hdu.close()
    else:
        print(f'No new tables required - already has {extra_tab_bytes} bytes of tables')

    
def fix(fname):
    '''
    Fixes the given filename
    gcount
    totallength
    then adds tables
    '''

    fix_gcount(fname)
    # make sure the file length is a multiple of FITS_BLOCK_SIZE
    fix_length(fname)

    fix_tables(fname)
        
    newsize = os.path.getsize(fname)
    print(f'File {fname} fixed. new size is {newsize}')
    hdu = fits.open(fname)
    hdu.info()
    hdu.close()




def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Fix UV fits files because not closed properly', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    for f in values.files:
        fix(f)
        
    

if __name__ == '__main__':
    _main()
