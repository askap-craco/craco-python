from astropy.io import fits
from astropy.time import Time
import numpy as np
import os


dtype_to_bitpix = {np.dtype('>i2'):16,
                   np.dtype('>i4'):32,
                   np.dtype('>f4'):-32,
                   np.dtype('>f8'):-64}


def create_header_for_image_data(fname, wcs, im_shape, dtype, kwargs):
    #Make the main part of the header first that describes the image
    '''
    header = {}
    header['SIMPLE'] = 'T'
    header['NAXIS'] = 3
    naxis1, naxis2 = im_shape
    header['NAXIS1'] = naxis1
    header['NAXIS2'] = naxis2
    header['NAXIS3'] = 1    #Leaving it as 1 to start with, but will be updated at the end
    header['BITPIX'] = dtype_to_bitpix.get(dtype)      
    '''

    fake_data = np.arange(im_shape[0] * im_shape[1] * 1, dtype=dtype).reshape(1, im_shape[0], im_shape[1])
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(header = header, data = fake_data)


    #Now add the other useful information to the header, which is not mandatory
    
    header = hdu.header
    now = Time.now()
    header['Time'] = now.tai.mjd
    header['EPOCH'] = 2000
    for key, val in kwargs.items():
        header[key] = val
    
    
    hdu.writeto(fname, overwrite = True)

    header = fits.getheader(fname)
    #import IPython
    #IPython.embed()
    #now convert the header to string
    header_str = header.tostring()
    assert len(header_str)%2880 == 0, f"Astropy didn't make a header with the proper length while creating the header-- len(header.tostring()) = len(header_str)"

    hdr_len = len(header_str)
    return header, hdr_len

def fix_length(fname):
    filesize = os.path.getsize(fname)
    with open(fname, 'ab') as fout:
        n_extra_bytes = 2880 - filesize % 2880
        if n_extra_bytes == 2880:
            n_extra_bytes = 0

        print(f'Current position {fout.tell()} writing {n_extra_bytes}')
        fout.write(bytes(n_extra_bytes))

    newsize = os.path.getsize(fname)
    print(f'Wrote {n_extra_bytes} to {fname} to make it from {filesize} {newsize}')


class ImageHandler:
    def __init__(self, outname, wcs, im_shape, dtype, useful_info):
        self.header, self.header_len = create_header_for_image_data(outname, wcs, im_shape, dtype, useful_info)
        #print(f"The header string is as follows:\n{self.header_str}")
        self.im_shape = im_shape
        self.fname = outname
        assert dtype in dtype_to_bitpix.keys(), f"Provided dtype needs to be one of the following - {dtype_to_bitpix.keys()}"
        self.dtype = dtype
        self.fout = open(self.fname, 'r+b')
        self.fout.seek(self.header_len, 0)
        #self.fout.write(bytes(self.header_str, 'utf-8'))
        self.frames_written = 0

    def put_new_frames(self, frames, fout = None):
        '''
        Adds the given block of images to the fits file

        frames: list of 2-D arrays (images)
        '''
        assert [self.im_shape == frames[i].shape for i in range(len(frames))], f"Did not get the expected frame shape - {self.im_shape}"
        assert [self.dtype == frames[i].dtype for i in range(len(frames))], f"Dtype don't match!"


        for ii, frame in enumerate(frames):
            data = frame.astype(self.dtype)
            #print(data, data.dtype)
            data.tofile(self.fout)
            #self.fout.write(data.tobytes())            ##Try frame.tofile(self.fout)
            self.frames_written += 1
        #print(f"Written {self.frames_written} to the fits file on disk")

    def close(self):
        self.fout.close()
        self.fix_file_after_closing()

    def fix_file_after_closing(self):
        fix_length(self.fname)
        self.header['NAXIS3'] = self.frames_written
        hdr_str = self.header.tostring()
        assert len(hdr_str)%2880 == 0, f"Astropy didn't make a header with the proper length while fixing the header -- len(hdr.tostring()) = len(hdr_str)"

        with open(self.fname, 'r+b') as fout:
            fout.seek(0, 0)
            fout.write(bytes(hdr_str, 'utf-8'))
            assert fout.tell() % 2880 == 0, "File pointer not at a multiple of 2880 after writing the fixed header"
            fout.flush()

        












    


