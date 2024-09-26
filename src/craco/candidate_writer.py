import numpy as np
import os
from npy_append_array import NpyAppendArray as npa
import gzip
from astropy.time import Time
from astropy import units as u
from craft import fdmt

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

def compute_dm_width_scaling(max_dm_samps, max_boxcar, freqs):
    '''
    Computes the correct DM-width scaling using the FDMT class (get_eff_var_recursive)
    Provides a correction factor multiplier after accounting for the fact that the search pipeline
    has already applied a sqrt(ibox * nchan) scaling.

    Arguments
    ---------
    max_dm_samps: int
                Maximum dm trial in sample units

    max_boxcar: int
                Maximum boxcar width trial in sample units

    freqs: np.ndarray
                Array containing the channel frequencies in Hz floats  
    '''
    nf = len(freqs)
    chw = freqs[1] - freqs[0]
    f_min = freqs[0] - chw/2
    thefdmt = fdmt.Fdmt(f_min=f_min, f_off = chw, n_f = nf, max_dt = max_dm_samps+1, n_t = 256)    #I don't think the value of nt matters here, even if it is wrong
    eff_vars = np.zeros((thefdmt.max_dt, max_boxcar))
    applied_vars = np.zeros((thefdmt.max_dt, max_boxcar))
    
    for idm in range(thefdmt.max_dt):
        for ibox in range(max_boxcar):
            eff_vars[idm, ibox] = thefdmt.get_eff_var_recursive(idm, ibox+1)       #for thefdmt ibox=0 has width=1, so passing ibox+1
            applied_vars[idm, ibox] = nf * (ibox + 1)
    
    eff_sigma = np.sqrt(eff_vars)
    applied_sigma = np.sqrt(applied_vars)
    snr_multiplier = applied_sigma / eff_sigma

    assert np.all(snr_multiplier <= 1), "DM-width scaling went wrong somewhere! Blame VG!"

    return snr_multiplier


class CandidateWriter:
    raw_dtype = np.dtype([('snr', '<i2'), ('loc_2dfft', '<u2'), ('boxc_width', 'u1'), ('time', 'u1'), ('dm', '<u2')])
    raw_dtype_formats = ['g', 'g', 'g', 'g', 'g']

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
    out_dtype = np.dtype(out_dtype_list)

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
            '+.5f',  #dec_deg
            'g', # ibeam
            '.1f' # latency
        ]

    # dtype without beam and latency
    out_dtype_short = np.dtype(out_dtype_list[:14])

    def __init__(self, outname, freqs= None, max_dm_samps=None, max_boxcar_width=None, first_tstart=None, overwrite = True, delimiter = "\t", ibeam=0):
        '''
        Initialises the object, opens file handler and writes the header (if appropriate)

        outname: str
                Path to the output file. If it ends in '.npy' it will make a binary file
                Otherwise '.txt' for human readable


        freqs: np.array | optional
                Numpy array that contains frequencies of all channels in Hz
        
        max_dm_samps: int | optional
                Maximum DM searched in sample units

        max_boxcar_width: int | optional
                Maximum boxcar trial searched in sample units

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
        self.ibeam = ibeam
        self.make_string_formatter()
        if max_dm_samps is None and max_boxcar_width is None and freqs is None:
            self.snr_multiplier = None
        else:
            self.snr_multiplier = compute_dm_width_scaling(max_dm_samps, max_boxcar_width, freqs)
            
        self.open_files()
        


    def make_string_formatter(self):
        string_formatter = ""
        assert len(self.out_dtype) == len(self.out_dtype_formats), f'{len(self.out_dtype)} != {len(self.out_dtype_formats)}'
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
            self.fout.flush()
                
    def interpret_cands(self, rawcands, iblk, plan, raw_noise_level, candbuf=None):
        ncands = len(rawcands)
        first_tstart = self.first_tstart
        

        if self.snr_multiplier is None:
            raise RuntimeError(f"DM-width scaling correction factor not provided!")
        
        #Get the corrected snrs and remove all below 6 sigma
        true_snr = rawcands['snr'] * 1./raw_noise_level * self.snr_multiplier[rawcands['dm'], rawcands['boxc_width']]
        rawcands = rawcands[true_snr >= 6]

        #new ncands
        orig_ncands = ncands
        ncands = len(rawcands)

        if candbuf is None:
            candidates = np.zeros(ncands, self.out_dtype)
        else:
            assert candbuf.dtype == self.out_dtype
            candidates = candbuf[:ncands]
        
        # don't bother computing everything it if it's empty
        # also location2pix fails as you cant verctorize on size 0 inputs
        if ncands == 0:
            return candidates
        
        #candidates['hw_ncands'] = orig_ncands
        candidates['snr'] = true_snr[true_snr >= 6]
        location = rawcands['loc_2dfft']
        candidates['lpix'], candidates['mpix'] = location2pix(location, plan.npix)
        candidates['rawsn'] = rawcands['snr']
        candidates['time'] = rawcands['time']
        candidates['dm'] = rawcands['dm']
        candidates['boxc_width'] = rawcands['boxc_width']
        candidates['total_sample'] = iblk * plan.nt + rawcands['time']
        candidates['iblk'] = iblk
        tsamp_s = plan.tsamp_s.value
        candidates['obstime_sec'] = candidates['total_sample'] * tsamp_s
        
        # must set obstime_sec *BEFORE* calculating cand times
        cand_times = self.calc_cand_times(candidates) # Astropy times to calculate latency

        candidates['mjd'] = cand_times.utc.mjd
        dmdelay_ms = rawcands['dm'] * tsamp_s * 1e3
        candidates['dm_pccm3'] = dmdelay_ms / DM_CONSTANT / ((plan.fmin/1e9)**-2 - (plan.fmax/1e9)**-2)
        coords = plan.wcs.pixel_to_world(candidates['lpix'], candidates['mpix'])
        candidates['ra_deg'] = coords.ra.deg
        candidates['dec_deg'] = coords.dec.deg
        candidates['ibeam'] = self.ibeam

        candidates = self.update_latency(candidates, cand_times)

        return candidates
    
    def update_latency(self, candidates, cand_times=None, now=None):
        if now is None:
            now = Time.now()

        if cand_times is None:
            cand_times = self.calc_cand_times(candidates)

        candidates['latency_ms'] = (now - cand_times).to(u.millisecond)
        return candidates
    
    def calc_cand_times(self, candidates):
        '''
        Returns astropy Time so we can calculate latency
        '''
        # Old code: candidates['mjd'] = first_tstart.utc.mjd + candidates['obstime_sec'].astype(self.out_dtype['mjd']) / 3600 / 24
        cand_times = self.first_tstart + candidates['obstime_sec']*u.second
        return cand_times


    
    def write_cands(self, candidates):
        if self.outtype == 'npy':
            self.fout.append(candidates)
        elif self.outtype == 'txt':
            for candrow in candidates:
                self.fout.write(self.string_formatter.format(*candrow))

            self.fout.flush()
        else:
            raise Exception("VG has messed up somewhere! Kill him!")

    def interpret_and_write_candidates(self, rawcands, iblk, plan, raw_noise_level, candbuf=None):
        c = self.interpret_cands(rawcands, iblk, plan, raw_noise_level, candbuf)
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
