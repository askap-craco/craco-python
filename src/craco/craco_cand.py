# new craco candidate class for the new data strcuture
### data io related
from craco import uvfits_meta
from craft import uvfits
### data process related
from craft import craco_plan, sigproc
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

import matplotlib.pyplot as plt
import matplotlib as mpl

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_filterbank(filpath, tstart, ntimes):
    f = sigproc.SigprocFile(filpath)
    nelements = ntimes*f.nifs*f.nchans
    f.seek_data(f.bytes_per_element*tstart)

    if (f.nbits == 8): dtype = np.uint8
    elif (f.nbits == 32): dtype = np.float32

    v = np.fromfile(f.fin, dtype=dtype, count=nelements )
    v = v.reshape(-1, f.nifs, f.nchans)
    return v

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
        # packet drop will break this function...
        pass
        # nonnan_t = (~np.isnan(newtf)).sum(axis=0) != 0
        # newtf = newtf[:, nonnan_t]
        # trange = trange[nonnan_t]
    else:
        nonnan_t = (np.isnan(newtf)).sum(axis=0) == 0
        _index = np.arange(newtf.shape[-1])
        _index = _index[nonnan_t]
        trange = trange[_index[0]:_index[-1]+1]
        newtf = newtf[:, _index[0]:_index[-1]+1]
        
    return newtf, (trange[0], trange[-1])


def _get_overlap_index(value1, value2):
    """
    get overlaid index based on two value ranges
    
    for example, we are provided with value1 (10, 15) (both inclusive),
    value2 (11, 13). We want to know the overlapping region indices...
    
    #index1  0   1   2   3   4   5
    value1: 10, 11, 12, 13, 14, 15
    value2:     11, 12, 13
    #index2      0   1   2
    
    We want to get (1, 3) and (0, 2)
    
    Params
    ----------
    range1, range2: 2 elements of tuples
        starting and ending index of two series
        
    Returns
    ----------
    range1_, range2_: 2 elements of tuples
        overlaid index in the range1 and range2
    """
    # get overlapping value first...
    overlap_value = (max(value1[0], value2[0]), min(value1[1], value2[1]),)
    
    ### 
    range1_ = (overlap_value[0] - value1[0], overlap_value[1] - value1[0],)
    range2_ = (overlap_value[0] - value2[0], overlap_value[1] - value2[0],)
    
    return range1_, range2_

def _workout_slice_w_center(center, length, radius=5):
    """
    work out a slice to index given a radius and the whole length
    # length is used to make sure no indexerror is raised
    """
    return slice(max(center-radius, 0), min(center+radius+1, length))

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
        ### data flag option...
        pcbpath=None,
    ):
        if skip_blocks is not None:
            log.warning("`skip_blocks` not supported...")

        self.canduvfits = CandUvfits(
            uvfits=uvfits, metafile=metafile, flagant=flagant,
            flagchan=flagchan, start_mjd=start_mjd,
        ) ### all basic information are encoded in `self.canduvfits.plan`
        self.uvfits = uvfits; self.metafile = metafile 
        self.calfile = calfile; self.pcbpath = pcbpath

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

    def _get_pcb_mask(self,):
        if self.pcbpath is None: 
            self.pcbmask = None
        else:
            try:
                pcbdata = load_filterbank(
                    self.pcbpath, self.vtstart, 
                    self.vtend - self.vtstart + 1
                )
                self.pcbmask = (pcbdata == 0.).T
            except:
                self.pcbmask = None

    def extract_data(self, padding):
        ### get candidate/visibility time range
        self._get_cand_vis_range(padding = padding)
        self._get_pcb_mask() # get mask from phase center filterbank
        self.canduvfits.snippet(self.vtstart, self.vtend)

    ########## data manipulation ##########
    def process_data(
        self, plan=None, dm_pccm3=None, zoom_r = 5
    ):
        """
        calibrate, rotate, normalise data
        get filterbank, make images
        """
        self.datasnippet = DataSnippet(
            cracoplan = self.canduvfits.dataplan,
            uvsource = self.canduvfits.datauvsource,
            pcbmask = self.pcbmask,
        )

        ### get flagchan from self.canduvfits
        if self.canduvfits.flagchan is None:
            flagchan = ""
        flagchan = strrange(self.canduvfits.flagchan)

        if dm_pccm3 is None: dm_pccm3 = self.dm_pccm3

        odata = self.canduvfits.data.copy() # original data
        fdata = self.datasnippet.flag_chan(odata, flagchan) # data after channel flagging
        fdata = self.datasnippet.flag_pcb(fdata,)
        cdata = self.datasnippet.calibrate(fdata, self.calfile) # calibrated data
        ndata = self.datasnippet.normalise(cdata)
        ddata = self.datasnippet.dedisperse(ndata, dm=dm_pccm3)

        rdata = self.datasnippet.rotate(cdata, self.coord, self.canduvfits.datauvw)
        nrdata = self.datasnippet.normalise(rdata)

        ### get filterbank data
        self.filtb = self.datasnippet.get_filtb(nrdata) #this is zero dm

        ### get image data
        self.imgcube = self.datasnippet.image_data(ddata, plan=plan)

        ### get zoom in image
        _, llen, mlen = self.imgcube.shape
        self.imgzoomcube = self.imgcube[
            :, _workout_slice_w_center(self.mpix, mlen, zoom_r),
            _workout_slice_w_center(self.lpix, llen, zoom_r)
        ]

    ### make plots
    ### plot filterbank
    def plot_filtb(
        self, dm=0., keepnan=True, 
    ):
        filterbank_plot, trange_ = self.datasnippet.dedisp_filtb(
            filtb=self.filtb, dm=dm, keepnan=keepnan, 
            tstart=self.canduvfits.datarange[0],
        )
        filterbank_plot[filterbank_plot == 0.] = np.nan

        grid = mpl.gridspec.GridSpec(
            nrows=5, ncols=5, wspace=0., hspace=0.
        )
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(grid[1:, :-1])

        extent = [
            self.canduvfits.tsamp * trange_[0], # starting time
            self.canduvfits.tsamp * trange_[1], # ending time
            self.canduvfits.fmin / 1e6, 
            self.canduvfits.fmax / 1e6,
        ]

        ax1.imshow(
            filterbank_plot, 
            aspect="auto", origin="lower", 
            extent=extent, interpolation="none"
        )

        ax2 = fig.add_subplot(grid[0, :-1], sharex=ax1)
        ### this is something as a function of time
        t = np.linspace(trange_[0], trange_[1], trange_[1]-trange_[0]+1) * self.canduvfits.tsamp
        tmin = np.nanmin(filterbank_plot, axis=0)
        tmax = np.nanmax(filterbank_plot, axis=0)
        tmea = np.nanmean(filterbank_plot, axis=0)
        ax2.plot(t, tmin, color="C0")
        ax2.plot(t, tmax, color="C1")
        ax2.plot(t, tmea, color="C2")
        ax2.tick_params(labelbottom=False)

        ax3 = fig.add_subplot(grid[1:, -1], sharey=ax1)
        ax3.tick_params(labelleft=False)
        ### this is something as a function of freq
        f = np.linspace(
            self.canduvfits.fmin / 1e6, 
            self.canduvfits.fmax / 1e6, 
            self.canduvfits.nchan,
        )
        fmin = np.nanmin(filterbank_plot, axis=1)
        fmax = np.nanmax(filterbank_plot, axis=1)
        fmea = np.nanmean(filterbank_plot, axis=1)
        ax3.plot(fmin, f, color="C0")
        ax3.plot(fmax, f, color="C1")
        ax3.plot(fmea, f, color="C2")

        return fig, (ax1, ax2, ax3)

    ### butterfly plot
    def _calculate_dmt(self, dmfact=1e2, ndm=100):
        dm_step = self.dm_pccm3 / dmfact
        dmrange = self.dm_pccm3 + np.arange(-ndm // 2, ndm // 2) * dm_step
        ndm = dmrange.shape[0]

        trange = (
            self.canduvfits.datarange[0],
            2 * self.canduvfits.datarange[1] - self.canduvfits.datarange[0]
        )
        nt = trange[1] - trange[0] + 1

        dmt = np.zeros((ndm, nt)) * np.nan
        for idm, dm in enumerate(dmrange):
            filtb_, trange_ = self.datasnippet.dedisp_filtb(
                filtb=self.filtb, dm=dm, keepnan=True,
                tstart=self.canduvfits.datarange[0],
            )
            tf_ts = np.nanmedian(filtb_, axis=0)
            range1_, range2_ = _get_overlap_index(trange, trange_)

            dmt[idm, range1_[0]:range1_[1]+1] = tf_ts[range2_[0]:range2_[1]+1]
        return dmt, dmrange, trange

    def plot_dmt(self, dmfact=1e2, ndm=100):
        dmt, dm_range, trange = self._calculate_dmt(dmfact=dmfact, ndm=ndm)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)

        extent = (
            trange[0] * self.canduvfits.tsamp, 
            trange[1] * self.canduvfits.tsamp,
            dm_range[0], dm_range[-1]
        )

        ax.imshow(
            dmt, aspect="auto", extent=extent, 
            interpolation=None, origin="lower",
        )

        ax.scatter(
            self.total_sample * self.canduvfits.tsamp, 
            self.dm_pccm3,
            marker="X", s=200, fc="none", ec="black"
        )

        ax.set_xlabel("Time after the observation (s)")
        ax.set_ylabel("Despersion Measure (pc cm^-3)")

        return fig, ax

    ### plot field images
    @property
    def image_end_index(self):
        dets = self.total_sample
        viss = self.canduvfits.datarange[0]
        return dets - viss

    @property
    def image_start_index(self):
        return self.image_end_index - self.boxc_width

    def plot_diagnostic_images(self, vmin=None, vmax=None):
        medimg = np.nanmedian(self.imgcube, axis=0)
        stdimg = np.nanstd(self.imgcube, axis=0)

        fig = plt.figure(figsize=(12, 4))

        projection=self.canduvfits.dataplan.wcs
        #detection image
        ax = fig.add_subplot(1, 3, 1, projection=projection)

        detimg = self.imgcube[
            self.image_start_index:self.image_end_index+1
        ]
        
        ax.imshow(
            np.nanmean(detimg, axis=0), vmin=vmin, vmax=vmax, 
            origin="lower", aspect="auto",
        )
        ax.set_title("detect image")

        ax = fig.add_subplot(1, 3, 2, projection=projection)
        ax.imshow(
            medimg, vmin=None, vmax=None, 
            origin="lower", aspect="auto",
        )
        ax.set_title("median image")

        ax = fig.add_subplot(1, 3, 3, projection=projection)
        ax.imshow(
            np.log10(stdimg), vmin=None, vmax=None, 
            origin="lower", aspect="auto",
        )
        ax.set_title("std image (log10)")

        return fig

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
        self.dataplan = self._load_plan(self.datauvsource)

        datal = len(self.uvws)
        self.datauvw = self.uvws[datal // 2][0]

class DataSnippet:
    """
    class to manage uvfits snippet e.g., calibrate, get image etc.
    """
    def __init__(
        self, cracoplan, uvsource, pcbmask=None
    ):
        self.plan = cracoplan # ideally start at the burst time
        self.uvsource = uvsource
        self.pcbmask = pcbmask # this is for pipeline flagging purpose

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

    def flag_pcb(self, data, flagval=0.):
        fdata = data.copy()
        if self.pcbmask is None: return fdata
        ### sanity check first...
        nbl, nchan, npol, nt = fdata.shape
        mnchan, mnpol, mnt = self.pcbmask.shape
        if nt != mnt:
            logging.warning("time axis not matched!!!")
            return fdata
        ### now time to apply mask...
        for ibl in range(nbl):
            fdata.data[ibl][self.pcbmask] = flagval
            fdata.mask[ibl][self.pcbmask] = True
        return fdata

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
