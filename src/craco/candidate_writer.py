import numpy as np
import os
from npy_append_array import NpyAppendArray as npa
import gzip


# oooh, gosh - this is noughty
DM_CONSTANT = 4.15

def location2pix(location, npix=256):

    npix_half = npix//2
    
    vpix = (location//npix)%npix - npix_half
    if (vpix<0):
        vpix = npix+vpix
        
    upix = location%npix - npix_half
    if (upix<0):
        upix = npix+upix
        
    #location_index = ((npix_half+vpix)%npix)*npix + (npix_half+upix)%npix
    return vpix, upix

location2pix = np.vectorize(location2pix)

class CandidateWriter:
    raw_dtype = np.dtype([('snr', '<i2'), ('loc_2dfft', '<u2'), ('boxc_width', 'u1'), ('time', 'u1'), ('dm', '<u2')])
    raw_dtype_formats = ['g', 'g', 'g', 'g', 'g']

    out_dtype = np.dtype(
        [
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
            ('dec_deg', 'f4')
        ]
    )
    out_dtype_formats = \
        [
            '.1f',  #snr
            '3g',   #lpix
            '3g',   #mpix
            'g',    #boxc
            'g',    #time
            'g',    #dm
            'g',    #iblk
            'g',    #rawsn
            'g',    #total_sample
            '.4f',  #obstime_sec
            '.9f',  #mjd
            '.3f',  #dm_pccm3
            '.5f',  #ra_deg
            '+.5f'  #dec_deg
        ]

    def __init__(self, outname, first_tstart, overwrite = True, delimiter = "\t"):
        '''
        Initialises the object, opens file handler and writes the header (if appropriate)

        outname: str
                Path to the output file. If it ends in '.npy' it will make a binary file
                Otherwise '.txt' for human readable

        first_start: Time
                Astropy TIme for beginning of file
        
        overwrite: bool
                overwrite an existing file or not
        delimiter: str
                delimiter string to use if outtype=txt
                This is ignored if the outtype=bin
       
        '''
        self.outname = outname
        self.overwrite = overwrite
        self.first_tstart = first_tstart
        outtype = outname.split('.')[-1].lower()
        assert outtype in ('gz', 'txt','npy'), f'Invalid output type {outtype} for {outname}'
        self.gzip = False
        if outtype == 'gz':
            outtype = 'txt'
            assert self.outname.endswith('.txt.gz'), f'Invalid output file type {outname}'
            self.gzip = True
        
        self.outtype = outtype
        self.delimiter = delimiter
        self.make_string_formatter()
        self.open_files()

    def make_string_formatter(self):
        string_formatter = ""
        for ii in range(len(self.out_dtype)):
            string_formatter += f"{{{ii}:{self.out_dtype_formats[ii]}}}\t"

        self.string_formatter = string_formatter.strip() + "\n"         

    def open_files(self):
        self.fout = None
        if os.path.exists(self.outname):
            if self.overwrite:
                if not os.access(self.outname, os.W_OK):
                    raise IOError(f"File {self.outname} exists and I don't have write permissions")
                os.remove(self.outname)
            else:
                raise IOError(f"File {self.outname} exists, and overwrite = False")
        
        if self.outtype.lower() == 'txt':
            if self.gzip:
                self.fout = gzip.open(self.outname, 'wt')
            else:
                self.fout = open(self.outname, 'w')
        elif self.outtype.lower() == 'npy':
            self.fout = npa(self.outname)
        else:
            raise ValueError(f"Unknown outtype specified: {self.outtype}, expected - ['txt', 'npy']")
        
        self._write_header()

    def _write_header(self):
        if self.outtype == 'npy':
            #npa will write the appropriate header when it sees the first block
            pass
        else:
            hdr_str = self.delimiter.join(i for i in self.out_dtype.names)
            self.fout.write("# " + hdr_str + "\n")
                
    def interpret_cands(self, rawcands, iblk, plan, raw_noise_level):
        ncands = len(rawcands)
        first_tstart = self.first_tstart
        candidates = np.zeros(ncands, self.out_dtype)

        # don't bother computing everything it if it's empty
        # also location2pix fails as you cant verctorize on size 0 inputs
        if ncands == 0:
            return candidates

        location = rawcands['loc_2dfft']
        candidates['lpix'], candidates['mpix'] = location2pix(location, plan.npix)
        candidates['rawsn'] = rawcands['snr']
        candidates['time'] = rawcands['time']
        candidates['dm'] = rawcands['dm']
        candidates['boxc_width'] = rawcands['boxc_width']
        candidates['snr'] = rawcands['snr'] * 1./raw_noise_level
        candidates['total_sample'] = iblk * plan.nt + rawcands['time']
        candidates['iblk'] = iblk
        tsamp_s = plan.tsamp_s.value
        candidates['obstime_sec'] = candidates['total_sample'] * tsamp_s
        candidates['mjd'] = first_tstart.utc.mjd + candidates['obstime_sec'].astype(self.out_dtype['mjd']) / 3600 / 24
        dmdelay_ms = rawcands['dm'] * tsamp_s * 1e3
        candidates['dm_pccm3'] = dmdelay_ms / DM_CONSTANT / ((plan.fmin/1e9)**-2 - (plan.fmax/1e9)**-2)
        coords = plan.wcs.pixel_to_world(candidates['lpix'], candidates['mpix'])
        candidates['ra_deg'] = coords.ra.deg
        candidates['dec_deg'] = coords.dec.deg

        return candidates
    
    def write_cands(self, candidates):
        if self.outtype == 'npy':
            self.fout.append(candidates)
        elif self.outtype == 'txt':
            for candrow in candidates:
                self.fout.write(self.string_formatter.format(*candrow))

            self.fout.flush()
        else:
            raise Exception("VG has messed up somewhere! Kill him!")

    def interpret_and_write_candidates(self, rawcands, iblk, plan, raw_noise_level):
        c = self.interpret_cands(rawcands, iblk, plan, raw_noise_level)
        self.write_cands(c)
        return c

    def write_log(self, logline):
        '''
        Write a log entry
        Only supports 'txt' output type
        '''
        if self.outtype == 'txt':
            self.fout.write(logline.strip() + '\n')

    def dump_raw_cands(self, raw_cands, iblk):
        outname = f"raw_candidates_{iblk}.npy"
        np.save(outname, raw_cands)


    def close(self):
        if self.fout is not None:
            self.fout.close()
            self.fout = None

    __del__ = close
