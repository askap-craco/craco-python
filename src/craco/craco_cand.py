# new craco candidate class for the new data strcuture
### data io related
from craco import uvfits_meta
from craft import uvfits
### data process related
from craft import craco_plan
from craft.craco_kernels import Gridder, Imager 
from craft.craco import bl2ant, bl2array, pointsource, coord2lm
from craco import preprocess

from craft.cmdline import strrange

import numpy as np
import pandas as pd 
import copy

from astropy.time import Time 
from astropy.coordinates import SkyCoord
from astropy import units

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# for dedispersion
def calculate_dm_tdelay(f1, f2, dm):
    """
    calculate time delay due to the dispersion
    dt = 4.15 * DM * (f1**-2 - f2**-2)
    
    Params
    ----------
    f1, f2: float
        Two frequencies in Hz
    
    Return
    ----------
    dt: float
        time difference in s
    """
    f1 = f1 / 1e9; f2 = f2 / 1e9 # convert the value to GHz
    return 4.15 * dm * (f1**-2 - f2**-2) * 1e-3 #conver to s

def filterbank_roll(tf, dm, freqs, tint, tstart=0, keepnan=True):
    """
    dedisperse filterbank data with a given dm (part of code is stolen from plot_allbeams)

    Params
    ----------
    tf: numpy.ndarray
        filterbank data, shape is (nchan, nf)
    dm: float
        dispersion measure to be used in dedispersion
    freqs: numpy.ndarray
        a 1D array with all frequencies information
    tint: float
        integration (sampling) time for each timestamp
    tstart: int
        starting timestamp index... 0 by default

    Returns
    ----------
    newtf: numpy.ndarray
        dedispersed filterbank data
    trange: 2 element tuple
        start time index and end time index
    """
    nchan, nt = tf.shape
    assert tf.shape[0] == freqs.shape[0], "not equal number of channel in `tf` and `freqs`..."

    # put nan arrays in on the two side to avoid mismatch when performing np.roll
    tdelay_max = abs(calculate_dm_tdelay(freqs[0], freqs[-1], dm))
    shift_max = int(np.round(tdelay_max / tint))

    nanarr = np.ones((nchan, shift_max)) * np.nan
    tf_ = np.concatenate([nanarr, tf, nanarr], axis=-1)
    newtf = np.empty_like(tf_)

    reffreq = min(freqs)
    for f, freq in enumerate(freqs):
        tdelay = calculate_dm_tdelay(reffreq, freq, dm)
        shift = int(np.round(tdelay / tint))

        newtf[f, :] = np.roll(tf_[f, :], shift, axis=0)
        
    ### calculate the range of the new time axis...
    trange = np.arange(tstart - shift_max, tstart + nt + shift_max) # both included...
    
    ### perform clipping... remove nan values...
    if keepnan:
        nonnan_t = (~np.isnan(newtf)).sum(axis=0) != 0
        newtf = newtf[:, nonnan_t]
        trange = trange[nonnan_t]
    else:
        nonnan_t = (np.isnan(newtf)).sum(axis=0) == 0
        _index = np.arange(newtf.shape[-1])
        _index = _index[nonnan_t]
        trange = trange[_index[0]:_index[-1]+1]
        newtf = newtf[:, _index[0]:_index[-1]+1]
        
    return newtf, (trange[0], trange[-1])

class Cand:
    """
    craco candidate class

    include basic derived information for candidates
    """
    def __init__(
        self, 
        ### candidate related information
        ra_deg=None, dec_deg=None, dm_pccm3=None, total_sample=None, 
        boxc_width=0, lpix=128, mpix=128,
        ### data related information
        uvfits=None, metafile=None, calfile=None, 
        ### data load option
        start_mjd=None, skip_blocks=None,
        ### data flag option
        flagant=None, flagchan=None, 
    ):
        if skip_blocks is not None:
            log.warning("`skip_blocks` not supported...")

        self.canduvfits = CandUvfits(
            uvfits=uvfits, metafile=metafile, flagant=flagant,
            flagchan=flagchan, start_mjd=start_mjd,
        ) ### all basic information are encoded in `self.canduvfits.plan`
        self.uvfits = uvfits; self.metafile = metafile 
        self.calfile = calfile

        ### put all information to the attribute
        self.ra_deg, self.dec_deg = ra_deg, dec_deg
        self.dm_pccm3 = dm_pccm3; self.total_sample = total_sample
        self.boxc_width = boxc_width
        self.lpix, self.mpix = lpix, mpix

    @property
    def coord(self):
        return SkyCoord(self.ra_deg, self.dec_deg, unit=units.degree)

    ########## data extraction ##########
    def _get_cand_vis_range(self, padding=100):
        """
        get candidate visibility range just based on the information provided
        """
        fmin = self.canduvfits.fmin
        fmax = self.canduvfits.fmax
        if fmin > fmax: fmin, fmax = fmax, fmin # swap if necessary

        # calculate time delay due to the dispersion
        dt_dis = np.ceil(calculate_dm_tdelay(fmin, fmax, self.dm_pccm3) // self.canduvfits.tsamp)
        self.ctend = self.total_sample
        self.ctstart = self.ctend - dt_dis - self.boxc_width
        # these are for the candidate time range

        self.vtstart = int(self.ctstart - padding)
        self.vtend = int(self.ctend + padding)
        # these are for the visibility time range

    def extract_data(self, padding):
        ### get candidate/visibility time range
        self._get_cand_vis_range(padding = padding)
        self.canduvfits.snippet(self.vtstart, self.vtend)

    ########## data manipulation ##########
    def process_data(
        self, plan=None, dm_pccm3=None, 

    ):
        """
        calibrate, rotate, normalise data
        get filterbank, make images
        """
        self.datasnippet = DataSnippet(
            cracoplan = self.canduvfits.plan,
            uvsource = self.canduvfits.datauvsource
        )

        ### get flagchan from self.canduvfits
        if self.canduvfits.flagchan is None:
            flagchan = ""
        flagchan = strrange(flagchan)

        if dm_pccm3 is None: dm_pccm3 = self.dm_pccm3

        odata = self.canduvfits.data.copy() # original data
        fdata = self.datasnippet.flag_chan(odata, flagchan) # data after channel flagging
        cdata = self.datasnippet.calibrate(fdata, self.calfile) # calibrated data
        ndata = self.datasnippet.normalise(cdata)
        ddata = self.datasnippet.dedisperse(ndata, dm=dm_pccm3)

        rdata = self.datasnippet.rotate(cdata, self.coord, self.canduvfits.datauvw)
        nrdata = self.datasnippet.normalise(rdata)

        ### get filterbank data
        self.filtb = self.datasnippet.get_filtb(nrdata) #this is zero dm

        ### get image data
        self.imgcube = self.datasnippet.image_data(ddata, plan=plan)

class CandUvfits:
    def __init__(
        self, uvfits, metafile=None, 
        flagant=None, flagchan=None, start_mjd=None,
    ):
        ### all basic information from the input
        self.uvfits = uvfits
        self.metafile = metafile
        self.start_mjd = start_mjd
        self.flagant = flagant
        self.flagchan = flagchan
        ### load basic uvfits
        self.uvsource = self._load_uvfits(start_mjd=self.start_mjd, flagant=self.flagant)
        self.plan = self._load_plan(self.uvsource, )

    def _load_uvfits(self, skip_blocks=0, start_mjd=None, flagant=None,):
        """
        function to load uvfits file
        """
        if self.metafile is None:
            uvsource = uvfits.open(
                self.uvfits, start_mjd=start_mjd, skip_blocks=skip_blocks
            )
        else:
            uvsource = uvfits_meta.open(
                self.uvfits, metadata_file = self.metafile, 
                start_mjd=start_mjd, skip_blocks = skip_blocks,
            )
        
        ### flag antennas
        if flagant is not None:
            uvsource = uvsource.set_flagants(strrange(flagant))
        return uvsource

    def _load_plan(self, uvsource, values=""):
        return craco_plan.PipelinePlan(uvsource, values=values)

    @property
    def fmin(self):
        return self.plan.freqs[0]
    
    @property
    def fmax(self):
        return self.plan.freqs[-1]

    @property
    def nchan(self):
        return self.plan.freqs.shape[0]

    @property
    def foff(self):
        return (self.fmax - self.fmin) / (self.nchan - 1)

    @property
    def tsamp(self):
        return self.plan.tsamp_s.value

    def snippet(self, tstart, tend,):
        """
        use fast_time_block, extract uvfits data with tstart and tend
        tstart and tend are in the unit of sample from the start_mjd
        note - tend included
        """
        if tstart < 0: tstart = 0
        assert tend >= tstart, f"end sample {tend} should be larger than the start sample {tstart}"
        ### note 
        block_iter = self.uvsource.fast_time_blocks(
            istart = tstart, nt = 1, fetch_uvws = True,
        )

        _data = []; _uvw = []; _mask = []
        for i in range(tstart, tend + 1):
            try:
                data, uvw = block_iter.__next__()
                ### ask vivek what does each dimension mean
                data = np.squeeze(data)[..., None, None] # add pol, time axis
                _data.append(data.data); _mask.append(data.mask)
                _uvw.append(uvw)
            except:
                log.info(f"cannot load full data... stops at {i}")
                tend = i - 1
                break

        if tend < tstart: 
            log.error("no data extracted... aborted...")
            raise ValueError("no available extracted...")
        
        ### concat data
        self.data = np.ma.array(
            np.concatenate(_data, axis=-1),
            mask = np.concatenate(_mask, axis=-1)
        )
        self.uvws = _uvw

        self.datarange = (tstart, tend) #tend included

        ### load another uvsource
        rawskip = self.uvsource.skip_blocks
        dataskip = (tstart + tend) // 2

        self.datauvsource = self._load_uvfits(
            skip_blocks = rawskip + dataskip,
            flagant = self.flagant,
        )

        datal = len(self.uvws)
        self.datauvw = self.uvws[datal // 2][0]

class DataSnippet:
    """
    class to manage uvfits snippet e.g., calibrate, get image etc.
    """
    def __init__(
        self, cracoplan, uvsource
    ):
        self.plan = cracoplan # ideally start at the burst time
        self.uvsource = uvsource

    ### flagging data
    def flag_chan(self, data, chans, flagval=0):
        """
        flag channels
        """
        # note preprocess.rfi_cleaner.flag_chans does not work...
        fdata = data.copy() # copy the data instead of changing the original data
        fdata.data[:, chans, ...] = flagval
        fdata.mask[:, chans, ...] = True # shape of data, nbl, nchan, npol, nt
        return fdata


    def flag_freq(self, data, freqs):
        raise NotImplementedError("flag frequency not implemented...")

    ### calibration
    def calibrate(self, data, calfile=None):
        if calfile is None:
            return data.copy()
        calibrator = preprocess.Calibrate(
            plan = self.plan, block_dtype = type(data),
            miriad_gains_file = calfile,
            baseline_order = self.plan.baseline_order
        )

        return calibrator.apply_calibration(data.copy())

    ### visibility rotation
    def rotate(self, data, coord, uvw):
        """
        rotate visibility to a given coordinate
        """
        lm = coord2lm(coord, self.plan.phase_center)
        psvis = pointsource(
            amp = 1, lm = lm, freqs = self.plan.freqs,
            baselines = uvw, baseline_order = self.plan.baseline_order
        )

        return data * np.conj(psvis)[..., None, None]

    ### normalise data
    def normalise(self, data, target_input_rms=1):
        return preprocess.normalise(
            data.copy(), target_input_rms = target_input_rms
        )

    ### dedisperse data
    def dedisperse(self, data, dm=0., target_input_rms=1):
        dedisperser = preprocess.Dedisp(
            freqs=self.plan.freqs, tsamp=self.plan.tsamp_s.value,
            dm_pccc = dm,
        )

        ddata = dedisperser.dedisperse(0, data)

        ### note - we need to make sure it is an even number
        nbl, nchan, npol, nt = ddata.shape
        if nt % 2 != 0: ddata = ddata[..., :-1]
        return ddata[..., 0, :]

    ######## filterbank related #########
    def get_filtb(self, block):
        nbl, nchan, npol, nt = block.shape
        assert npol == 1, "cannot handle multiple polarization now..."
        filtb = np.nanmean(block, axis = 0)[:, 0, :].real

        return filtb # shape of the data nchan, nt

    def dedisp_filtb(self, filtb, dm=0., tstart=0, keepnan=True):
        """
        dedisperse the filterbank data, tstart is the starting time of the data
        """
        return filterbank_roll(
            tf = filtb, dm = dm, freqs = self.plan.freqs,
            tint = self.plan.tsamp_s.value, 
            tstart = tstart, keepnan = keepnan,
        )

    ######### Synthesized images ##########
    def _image_grid_data(self, imager, gdata):
        return imager(np.fft.fftshift(gdata)).astype(np.complex64)

    def _prepare_image(self, uvsource=None, plan=None):
        if uvsource is None: uvsource = self.uvsource 
        if plan is None: plan = self.plan

        gridder = Gridder(uvsource, plan, plan.values)
        imager = Imager(uvsource, plan, plan.values)
        return gridder, imager

    def image_data(self, data, uvsource=None, plan=None):
        """
        make synthesized image with dedispersed data

        note - if you want to make a larger image, update plan value here
        """
        gridder, imager = self._prepare_image(uvsource, plan)
        gdata = gridder(data) # with a shape if (npix, npix, nt)
        cidata = np.array([
            self._image_grid_data(imager, gdata[..., i])
            for i in range(gdata.shape[-1])
        ]) # complex image data

        idata = []
        for i in cidata:
            idata.append(i.real[None, ...])
            idata.append(i.imag[None, ...])
        return np.concatenate(idata)
