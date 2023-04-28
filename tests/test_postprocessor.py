from numpy import testing
import unittest, tempfile, os
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from craco.postprocess import create_header_for_image_data, dtype_to_bitpix, ImageHandler

'''
def get_random_filename():
    randn = str(np.random.randint(0, 100000)).zfill(5)
    outname = "./test_file_" + randn + ".tmp"

def test_module_import():
    try:
        from craco import postprocess
    except ImportError as IE:
        print(f"Could not import postprocess from craco due to\n{IE}")


from craco import postprocess

def test_create_header_for_image_data():
    fname = get_random_filename()
    w = WCS(naxis=2)
    w.wcs.crpix = [128.5, 128.5]
    w.wcs.cdelt = np.array([-0.0081, 0.008])
    w.wcs.crval = [180, -45]
    w.wcs.ctype = ["RA--SIN", "DEC--SIN"]
    w.wcs.cunit = ['deg', 'deg']
    w.wcs.set_pv(((2, 1, 0.1), (2, 2, -0.1)))

    im_shape = (256, 256)

    dtype = np.float32

    kwargs = {'TEST': 'test'}

    header, hdr_len = postprocess.create_header_for_image_data(fname, w, im_shape, dtype, kwargs)

    assert type(header) == dict
    assert type(hdr_len) == int
    assert os.path.exists(fname)
    filesize = os.path.getsize(fname)
    assert filesize%2880 == 0

    hdu = fits.open(fname)
    fits_header = hdu[0].header

    assert header == fits_header

    fits_data = hdu[0].data
    assert fits_data.shape == im_shape
    assert fits_data.dtype == dtype
'''



class TestCreateHeaderForImageData(unittest.TestCase):

    def test_create_header_for_image_data(self):
        # create FITS file
        wcs = WCS(naxis=2)
        im_shape = (10, 10)
        dtype = np.dtype('>f4')
        kwargs = {'OBSERVER': 'Tester'}
        fname = tempfile.NamedTemporaryFile(suffix='.fits').name

        header, hdr_len = create_header_for_image_data(fname, wcs, im_shape, dtype, kwargs)

        # load FITS file and check header values
        loaded_header = fits.getheader(fname)
        self.assertEqual(loaded_header['SIMPLE'], True)
        self.assertEqual(loaded_header['NAXIS'], 3)
        self.assertEqual(loaded_header['NAXIS1'], im_shape[0])
        self.assertEqual(loaded_header['NAXIS2'], im_shape[1])
        self.assertEqual(loaded_header['NAXIS3'], 1)
        self.assertEqual(loaded_header['BITPIX'], dtype_to_bitpix[dtype])
        self.assertEqual(loaded_header['OBSERVER'], 'Tester')

        # check header string length
        header_str = header.tostring()
        self.assertEqual(len(header_str) % 2880, 0)
        loaded_header_str = loaded_header.tostring()
        self.assertEqual(header_str, loaded_header_str)
        


        # check the data values
        expected_data = np.arange(im_shape[0] * im_shape[1] * 1, dtype=dtype).reshape(1, im_shape[0], im_shape[1])
        loaded_data = fits.getdata(fname)
        self.assertTrue(np.allclose(expected_data, loaded_data))

        # check data size
        loaded_data_size = loaded_data.itemsize * loaded_data.size
        expected_data_size = np.dtype(dtype).itemsize * im_shape[0] * im_shape[1]

        self.assertEqual( loaded_data_size, expected_data_size  )
        # check file size        
        #expected_file_size = expected_hdr_size + expected_data_size
        loaded_file_size = os.path.getsize(fname)
        print(f"The file size is {loaded_file_size}, the size%2880 = {loaded_file_size%2880}")
        #self.assertEqual(loaded_file_size, expected_file_size)

    def test_image_handler(self):
        wcs = WCS(naxis=2)
        im_shape = (256, 256)
        dtype = np.dtype('>f4')
        print(dtype)
        kwargs = {'OBSERVER': 'Tester'}

        fname = tempfile.NamedTemporaryFile(suffix='.fits').name
        
        img_handler = ImageHandler(outname = fname, wcs= wcs, im_shape=im_shape, dtype=dtype, useful_info=kwargs)

        test_image = np.random.normal(0, 1, im_shape[0] * im_shape[1]).astype(dtype).reshape(im_shape)
        a = test_image.copy()
        print(a.dtype)
        img_handler.put_new_frames([a])
        b = test_image + np.ones_like(test_image, dtype=dtype)
        print(b.dtype)
        img_handler.put_new_frames([b.astype(dtype)])
        c = test_image - np.ones_like(test_image, dtype=dtype)
        print(c.dtype)
        img_handler.put_new_frames([c.astype(dtype)])

        img_handler.close()

        loaded_data = fits.getdata(fname)

        print(loaded_data)
        #import IPython
        #IPython.embed()
        self.assertTrue(np.allclose(loaded_data, test_image))






if __name__ == '__main__':
    x = TestCreateHeaderForImageData()
    x.test_create_header_for_image_data()
    x.test_image_handler()


