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

def fix_length(fname):
    '''
    Appends zeros to the end of a fits file to make it a multiple of 2880 bytes
    '''
    filesize = os.path.getsize(fname)
    with open(fname, 'ab') as fout:
        n_extra_bytes = 2880 - filesize % 2880
        if n_extra_bytes == 2880:
            n_extra_bytes = 0

        print(f'Current position {fout.tell()} writing {n_extra_bytes}')
        fout.write(bytes(n_extra_bytes))

    newsize = os.path.getsize(fname)
    print(f'Wrote {n_extra_bytes} to {fname} to make it from {filesize} {newsize}')

    
def fix(fname, values):

    
    with open(fname+'.groupsize', 'r') as fin:
        groupsize = int(fin.read())
        
    filesize = os.path.getsize(fname)
    hdr = fits.getheader(fname)
    #hdr = fits.header.Header.fromtextfile(fname+'.header')
    gcount = hdr['GCOUNT'] 
    if gcount != 1:
        print(f'File {fname} already fixed with GCOUNT={gcount}. not fixing')
        hdu = fits.open(fname)
        hdu.info()
        hdu.close()
        return

    gcount = (filesize - len(hdr.tostring())) // groupsize
    hdr['GCOUNT'] = gcount
    hdr['FIXED'] = True
    print(f'File {fname} is size {filesize} - header = {len(hdr.tostring())} gcount={gcount}')
    with open(fname, 'r+b') as fout: # can't be 'a' as it only appends, irrepsective of seek position
        fout.seek(0,0)
        fout.write(bytes(hdr.tostring(), 'utf-8'))
        assert fout.tell() % 2880 == 0
        fout.flush()

    fix_length(fname)

    fq_table = fits.open(fname+'.fq_table')[1]
    an_table = fits.open(fname+'.an_table')[1]
    su_table = fits.open(fname+'.su_table')[1]
    hdu = fits.open(fname, 'append')
    #from IPython import embed
    #embed()
    hdu.append(fq_table)
    hdu.append(an_table)
    hdu.append(su_table)
    newsize = os.path.getsize(fname)
    print(f'File {fname} fixed. new size is {newsize}')
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
        fix(f, values)
        
    

if __name__ == '__main__':
    _main()
