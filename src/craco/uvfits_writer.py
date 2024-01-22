from astropy.io import fits
import numpy as np
from craco import fixuvfits


CRACO_DEFAULT_DATA_DTYPE_DICT = {
    'names': ['UU', 
              'VV',
              'WW',
              'DATE',
              'BASELINE',
              'FREQSEL',
              'SOURCE',
              'INTTIM',
              'DATA'],
              
    'formats':['>f4',
               '>f4',
               '>f4', 
               '>f4',
               '>f4',
               '>f4',
               '>f4',
               '>f4',
               '>f4']
}
CRACO_DTYPE = np.dtype(CRACO_DEFAULT_DATA_DTYPE_DICT)

class UvfitsReader:
    def __init__(self, infile):
        self.infile = infile
        self.fin = fits.open(self.infile)


class UvfitsWriter:

    def __init__(self, outname, infile = None):
        self.outname = outname
        self.uvr = None
        if infile:
            self.uvr = UvfitsReader(self.infile)
        self.open_outfile()
        self.gcount = 0
        pass

    def open_outfile(self):
        self.fout = open(self.outname, 'wb')

    def close_file(self, fix_length = True):
        self.fout.flush()
        self.fout.close()
        if fix_length:
            self.fix_length(self.outname)
        

    def write_header(self, header):
        self.fout.seek(0, 0)
        self.fout.write(bytes(header.tostring(), 'utf-8'))
    
    def write_block(self, block)