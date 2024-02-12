from astropy.io import fits
import numpy as np
from craco import fixuvfits, uvfits_meta
from craft import uvfits
import logging

log = logging.getLogger(__name__)

__author__ = "Vivek Gupta <vivek.gupta@csiro.au>"

CRACO_DATA_DEFAULT_SHAPE = (1, 1, 1, 288, 1, 3)
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
               ('>f4', CRACO_DATA_DEFAULT_SHAPE)]
}
CRACO_DTYPE = np.dtype(CRACO_DEFAULT_DATA_DTYPE_DICT)

def convert_blockbl_to_visdata(blockbl):
    '''
    Converts a data array for a single baseline at a single time
    into a visrow format data array. That is, it converts
    a complex (optionally masked) array of shape (nf, npol) into
    a real array of shape (1, 1, 1, nf, npol, 3)
    '''
    if blockbl.ndim == 2:
        nf, npol = blockbl.shape
    elif blockbl.ndim == 1:
        nf = blockbl.shape
        npol = 1
    else:
        raise ValueError(f"blockbl has invalid shape = {blockbl.shape}")
    
    visdata = np.empty((1, 1, 1, nf, npol, 3))
    visdata_element_shape = visdata.shape[:-1]
    if isinstance(blockbl, np.ma.core.MaskedArray):
        weights = 1 - blockbl.mask
    else:
        weights = np.ones_like(blockbl)

    visdata[..., 0] = blockbl.real.reshape(visdata_element_shape)
    visdata[..., 1] = blockbl.imag.reshape(visdata_element_shape)
    visdata[..., 2] = weights.reshape(visdata_element_shape)

    return visdata

def copy_visparams_to_visrow(visrows, UU, VV, WW, DATE, BASELINE, FREQSEL, SOURCE, INTTIM):
    assert visrows.size == UU.size, f"{visrows.size}, {UU.size}"
    visrows['UU'] = UU
    visrows['VV'] = VV
    visrows['WW'] = WW
    visrows['DATE'] = DATE
    visrows['BASELINE'] = BASELINE
    visrows['FREQSEL'] = FREQSEL
    visrows['SOURCE'] = SOURCE
    visrows['INTTIM'] = INTTIM
    return visrows


'''
class UvfitsReader:
    def __init__(self, infile):
        self.infile = infile
        self.fin = fits.open(self.infile)
        self.dtype = self.fin[0].data.dtype
        self.header = self.fin[0].header
'''

class UvfitsWriter:

    def __init__(self, outname, infile = None):
        self.outname = outname
        self.uvr = None
        self.header = None
        if infile:
            self.uvr = uvfits_meta.open(infile)
        else:
            raise NotImplementedError("-- :( --")
        self.open_outfile()
        pass

    @property
    def dtype(self):
        if self.uvr:
            return self.uvr.dtype
        else:
            return CRACO_DTYPE

    def open_outfile(self):
        self.fout = open(self.outname, 'wb')
        self.gcount = 0

    def close_file(self, fix_length = True):
        self.fout.flush()
        self.fout.close()
        if fix_length:
            fixuvfits.fix_length(self.outname)
        self.fout = None
        
    def write_header(self, header=None):
        if not header:
            if not self.header:
                raise ValueError("You need to either specify a header, or call self.copy_header() first")
            header = self.header
        
        self.fout.seek(0, 0)
        hdrb = bytes(header.tostring(), 'utf-8')
        assert len(hdrb) % 2880 == 0, f"{len(hdrb)}. self.header = \n{self.header}"
        self.fout.write(hdrb)
    
    def copy_header(self):
        if self.uvr is None:
            raise ValueError("Cant copy header without an infile")
        self.header = self.uvr.hdulist[0].header.copy()
        self.write_header()

    def append_supplementary_tables(self, uvsource:uvfits_meta.UvfitsMeta = None):
        log.info("Appending supplementary tables")
        if self.fout:
            self.close_file()

        if not uvsource:
            uvsource = self.uvr
            if not self.uvr:
                raise ValueError("I need a uvsource or an infile to be able to copy tables from")
        assert isinstance(uvsource, uvfits_meta.UvfitsMeta) or isinstance(uvsource, uvfits.UvFits), f"Uvsource is of type {type(uvsource)}"

        fout = fits.open(self.outname, 'append')

        for it, table in enumerate(uvsource.hdulist[1:]):
            row = table.data[0]
            if table.name == 'AIPS SU' and row['SOURCE'].strip() == 'UNKNOWN':
                row['SOURCE'] = uvsource.target_name
                row['RAEPO'] = uvsource.target_skycoord.ra.deg
                row['DECEPO'] = uvsource.target_skycoord.dec.deg
                print('Replaced UNKNOWN source with %s %s', uvsource.target_name, uvsource.target_skycoord.to_string('hmsdms'))
            fout.append(table)

        fout.flush()
        fout.close()

    def update_header(self, update_params_dict= None, update_gcount = True):
        gcount_updated = False
        if update_params_dict:
            for param, val in update_params_dict.items():
                self.header[param] = val
                if param == "GCOUNT":
                    gcount_updated = True
        
        if not gcount_updated and update_gcount:
            self.header['GCOUNT'] = self.gcount

    def write_visrows_to_disk(self, visrows):
        assert visrows.dtype == self.dtype, f"{visrows.dype}, {self.dtype}"
        nrows = len(visrows)
        log.debug(f"Dumping {nrows} visrows to disk")
        visrows.tofile(self.fout)
        self.gcount += visrows.size
        log.debug(f"Written {self.gcount} visrows so far")

    def write_time_block(self, block, UU, VV, WW, DATE, BASELINE, FREQSEL, SOURCE, INTTIM):
        nbl = block.shape[0]
        nt = block.shape[-1]
        nbl_nt = nbl * nt
        assert len(UU) == nbl_nt
        assert len(VV) == nbl_nt
        assert len(WW) == nbl_nt
        assert len(DATE) == nbl_nt
        assert len(BASELINE) == nbl_nt
        assert len(FREQSEL) == nbl_nt
        assert len(SOURCE) == nbl_nt
        assert len(INTTIM) == nbl_nt
        for it in range(nt):
            it_date = DATE[it*nbl]
            for ibl in range(nbl):
                it_ibl = it * nbl + ibl
                assert DATE[it_ibl] == it_date
                data = convert_blockbl_to_visdata(block[ibl, ..., it])
                visrow = np.array([(UU[it_ibl],
                                    VV[it_ibl],
                                    WW[it_ibl],
                                    DATE[it_ibl],
                                    BASELINE[it_ibl],
                                    FREQSEL[it_ibl],
                                    SOURCE[it_ibl],
                                    INTTIM[it_ibl],
                                    data)])
                self.write_visrows_to_disk(visrow)
