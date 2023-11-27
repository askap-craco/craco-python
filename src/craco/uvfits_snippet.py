from craco import uvfits_meta
import numpy as np

from astropy.io import fits
#from astropy.io.fits.fitsrec import FITS_rec

DTYPE2BITPIX = {'uint8': 8, 'int16': 16, 'uint16': 16, 'int32': 32,
                'uint32': 32, 'int64': 64, 'uint64': 64, 'float32': -32,
                'float64': -64, 'complex64':-32, 'complex128':-64}

def get_total_nsamps(uvsource):
    total_nsamps = uvsource.vis.size // uvsource.nbl
    return int(total_nsamps)

def copy_data_and_masks(new_data, desired_data):
    desired_data[:, 0, 0, 0, :, :, 0] = new_data.real
    desired_data[:, 0, 0, 0, :, :, 1] = new_data.imag
    if isinstance(new_data, np.ma.core.MaskedArray):
        desired_data[:, 0, 0, 0, :, :, 2] = 1 - new_data.mask.astype(int)

def make_parameter_cols(arr, uvws = None):
    parnames = []
    pardata = []
    parbzeros = []
    first_date = arr['DATE'][0]
    if uvws is not None:
        reshaped_uvws = uvws.transpose(1, 2, 0).reshape(3, -1)
    for parname in arr.dtype.names:
        if parname != "DATA":
            parnames.append(parname)
            
            if parname == 'DATE':
                pardata.append(arr[parname] - first_date)
                parbzeros.append(first_date)

            elif parname == 'UU' and uvws is not None:
                pardata.append(reshaped_uvws[0, :])
                parbzeros.append(0)

            elif parname == 'VV' and uvws is not None:
                pardata.append(reshaped_uvws[1, :])
                parbzeros.append(0)

            elif parname == 'WW' and uvws is not None:
                pardata.append(reshaped_uvws[2, :])
                parbzeros.append(0)
            else:
                pardata.append(arr[parname])
                parbzeros.append(0)

    return parnames, pardata, parbzeros

def makeGroupData(visrows):
    parnames, pardata, parbzeros= make_parameter_cols(visrows)
    GroupData = fits.GroupData(visrows['DATA'], parnames = parnames, pardata = pardata, bzero = 0.0, bscale = 1.0, parbzeros=parbzeros)
    return GroupData

class UvfitsSnippet:

    def __init__(self, uvpath, start_samp:int = 0, end_samp:int = 10, metadata_file=None):
        '''
        start_samp and end_samp are inclusive
        end_samp can be -1 to specify the end of the file
        '''
        self.uvpath = uvpath
        self.uvsource = uvfits_meta.open(self.uvpath, metadata_file=metadata_file)
        self.total_nsamps = get_total_nsamps(self.uvsource)
        self.nbl = self.uvsource.nbl

        if end_samp == -1:
            end_samp = self.total_nsamps - 1

        assert isinstance(start_samp, int) and start_samp <= self.total_nsamps
        assert isinstance(end_samp, int) and end_samp <= self.total_nsamps
        assert start_samp <= end_samp

        self.start_samp = start_samp
        self.end_samp = end_samp
        self.start_idx = start_samp * self.nbl
        self.end_idx = (end_samp + 1) * self.nbl

        self._GroupData = None

    def read_as_visrows(self):
        if self._GroupData is None:
            self._GroupData = makeGroupData(self.uvsource.vis[self.start_idx: self.end_idx])
            #self._GroupData = fits.GroupData(FITS_rec(self.uvsource.vis[self.start_idx: self.end_idx + 1]))

    def read_as_data_block_with_uvws(self):
        dout, uvwout, samp_ends =  self.uvsource.time_block_with_uvw_range((self.start_samp, self.end_samp))
        assert samp_ends == (self.start_samp, self.end_samp)
        return dout, uvwout

    @property
    def data(self):
        self.read_as_visrows()
        return self._GroupData

    def save(self, outname=None, overwrite=False):
        if outname is None:
            outname = self.uvpath.replace(".uvfits", f".t{self.start_samp}_{self.end_samp}.uvfits")

        header = self.uvsource.header
        #TODO - verify if the header GCOUNT gets updated. If not, then uncomment the following lines
        #header = self.uvsource.header.copy()
        #header['GCOUNT'] = len(self.data)
        GroupsHDU = fits.GroupsHDU(self.data, header = header)
        HDUList = fits.HDUList([GroupsHDU, *self.uvsource.hdulist[1:]])
        row = HDUList[3].data[0]
        row['SOURCE'] = self.uvsource.target_name
        row['RAEPO'] = self.uvsource.target_skycoord.ra.deg
        row['DECEPO'] = self.uvsource.target_skycoord.dec.deg

        HDUList.writeto(outname, overwrite=overwrite)


    def swap_with_visrows(self, new_data):
        '''
        It can swap the data with new_data.
        new_data needs to be an object of fits.GroupData class
        If you've got a uvfits.vis[slice] object, then just do fits.GroupData(FITS_rec(uvfits.vis[slice])) to convert it
        new_data may or may not have the same length (GCOUNT) as the old data!
        '''

        assert isinstance(new_data, fits.GroupData)

        if new_data.dtype == self.data.dtype:
            #I am assuming here that the new_data can be swapped in as is, even if it has more/less no. of rows than the original data
            self._GroupData = new_data

    def swap_with_data(self, new_data, new_uvws = None, parnames=None, pardata=None, bzero=0, bscale=1):
        '''
        It can swap the data with new_data.
        new_data can be np.ndarray or np.ma.core.MaskedArray
        new_uvws - numpy array containing uvw values shape = [nbl, 3, nt]

        You can provide arrays with any of the following types:
        shape (nbl * nt, 1, 1, 1, nf, npol, 3) -- output of uvfits.vis[slice]['DATA']
        shape (nbl * nt, nf, npol, 3) -- output of uvfits.vis[slice]['DATA'].squeeze()
        shape (nbl, nf, npol, nt) complex np.ndarray/masked_array -- output of bl2array(uvsource.time_blocks())
        '''


        gd_shape = self.data['DATA'].shape
        gd_squeezed_shape = self.data['DATA'].squeeze().shape
        gd_nonzero_ndim = self.data['DATA'].squeeze().ndim
        nf = gd_shape[-3]
        npol = gd_shape[-2]
        ncmplx = gd_shape[-1]
        nt_nbl = gd_shape[0]

        expected_complex_block_shape = (self.nbl, nf, npol, nt_nbl // self.nbl)

        assert DTYPE2BITPIX[new_data.dtype.name] == self.uvsource.header['BITPIX'], f"data dtype mismatch with bitpix in header, {new_data.dtype.name}-{DTYPE2BITPIX[new_data.dtype.name]} vs {self.uvsource.header['BITPIX']}"

        if new_data.shape == gd_shape:
            desired_data = new_data

        elif new_data.shape == gd_squeezed_shape:
            desired_data = np.zeros(gd_shape)
            copy_data_and_masks(new_data, desired_data)

        elif new_data.shape == expected_complex_block_shape:
            assert gd_nonzero_ndim != 4, f"Data axes besides nbl*nt, nf, npol, ncmplx are not empty! I won't know how to create those, new_data.shape = {new_data.shape}, expected_complex_block_shape = {expected_complex_block_shape}, gd_nonzero_ndim = {gd_nonzero_ndim}"
            desired_data = np.zeros(gd_shape, dtype=self.data['DATA'].dtype)
            new_data = new_data.transpose((3, 0, 1, 2)).reshape(-1, nf, npol)
            copy_data_and_masks(new_data, desired_data)

        else:
            raise ValueError(f"I expect new data to have shape - {gd_shape} or {gd_squeezed_shape} or {expected_complex_block_shape}. Given - {new_data.shape}")

        if parnames is None or pardata is None:
            parnames, pardata, parbzeros = make_parameter_cols(self.data, new_uvws)

        self._GroupData = fits.GroupData(desired_data,
                                       parnames = parnames,
                                       pardata = pardata,
                                       bzero = bzero,
                                       bscale = bscale, 
                                       parbzeros = parbzeros)

