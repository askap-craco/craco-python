#!/usr/bin/env python
# this script can be used as a package, and can be used as a script as well
from craft.craco import bl2ant, bl2array, pointsource, coord2lm
from craft.craco_kernels import Gridder, Imager 
from craco import preprocess
from craft import craco_plan
from craft import uvfits

import logging
import io 
import os
from PIL import Image

from astropy.coordinates import SkyCoord
import astropy.units as units
from astropy.io import fits

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


### function to auto flag the antenna...
def nbl2nant(nbl, auto=False):
    if auto:
        nant = (-1 + np.sqrt(1 + 8 * nbl))/2
    else:
        nant = (1 + np.sqrt(1 + 8 * nbl)) / 2
    return int(nant)

def find_flag(uvsource, uvmax=10000):
    """
    get flagged antenna automatically based on the uvmax...
    
    Param
    ----------
    uvsource: craft.uvfits
        uvfits file opened in craco...
    uvmax: float
        maximum uv threshold for determining bad antennas (in wavelengths)
        
    Retunrs
    ----------
    flagged antenna index (1-based) based on the uvmax

    Note
    ----------
    the returned value from this function can be used to pass to the set_flagant function
    """
    fmax = uvsource.channel_frequencies.max()
    nant = nbl2nant(uvsource.nbl)
    ant_flags = np.zeros(nant, dtype=int)
    
    # I don't care about the other values... 
    # it takes too long to get value from original dict...
    bl = uvsource.baselines
    uvw_dict = {
        blid: [bl[blid]["UU"], bl[blid]["VV"], bl[blid]["WW"]]
        for blid in bl
    }
    
    for blid in uvw_dict:
        # blid are numbers
        a1, a2 = bl2ant(blid) # 1-based
        umax = np.abs(uvw_dict[blid][0] * fmax)
        vmax = np.abs(uvw_dict[blid][1] * fmax)
        
        if umax > uvmax or vmax > uvmax:
            ant_flags[a1-1] += 1
            ant_flags[a2-1] += 1
            
    ### flag antenna where number of greater than have of the antenna...
    # antennas are 1-based...
    flagged_ants = np.arange(1, nant+1)[ant_flags >= nant // 2]
    logging.info(f"find {len(flagged_ants)} antennas flagged...")
            
    return list(flagged_ants)

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

def average_uvws(bluvws, metrics="mean"):
    """
    return single set of uvw values based on several sets of uvw vales

    Params
    ----------
    bluvws: dict, keys: baseline; value: numpy.ndarray (shape of (3, nt))
        baseline uvw values for a range of timestamps
    metrics: str, allowed values: mean, start, end
        how to average uvw values

    Return
    ----------
    bluvw: dict
        new dictionary for only one timestamp uvw values based on the metrics selected
    """
    assert metrics in ["mean", "start", "end"], "please select metrics from `mean`, `start` and `end`"

    if metrics == "mean":
        return {blid: bluvws[blid].mean(axis=-1, keepdims=True) for blid in bluvws}
    if metrics == "start":
        return {blid: bluvws[blid][:, [0]] for blid in bluvws}
    if metrics == "end":
        return {blid: bluvws[blid][:, [-1]] for blid in bluvws}

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
    
def importfig2img(fig):
    """
    without saving the matplotlib figure to disk,
    load it to PILLOW package directly
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches="tight")
    buffer.seek(0)
    
    plt.close() # close the image
    
    return Image.open(buffer)

def _vis2nbl(vis):
    """
    functions to infer the number of the baselines based on the visibility data from uvfits.
    This don't consider any flagging etc as we don't care about that
    """
    d0 = vis[0]["DATE"]

    nbl = 0
    for visrow in vis:
        d = visrow["DATE"]
        if d > d0: return nbl
        nbl += 1

### webpage related code
def _webpage_style():
    """
    define css here
    """
    return """<style>
    table.info {
        table-layout: fixed;
        width: 80%;
        border: 1px solid black;
        border-left: none; 
        border-right: none;
    }
    table.info tr{
        border: 1px solid black;
        border-left: none; 
        border-right: none;
    }
    table.info td {
        word-wrap: break-word;
        overflow-wrap: break-word;
        width: 25%;
    }

    table.threecol {
        table-layout: fixed;
        border: none;
        width: 90%;
    }
    table.threecol td {
        width: 33%;
    }

    table.fourcol {
        table-layout: fixed;
        border: none;
        width: 90%;
    }
    table.fourcol td {
        width: 25%;
    }
</style>
"""

def _webpage_info_table(infolst, ncol=2):
    """
    create a `ncol` table to show information of the burst
    """
    header = '''<table class="info" style="margin-left: auto; margin-right: auto;">\n'''
    body = ''
    for i in range(0, len(infolst), ncol):
        row = "<tr>\n"
        for j in range(ncol):
            if i + j >= len(infolst): 
                row += '''<td><b></b></td> <td></td>\n'''
            else:
                _key, _value = infolst[i+j]
                row += '''<td><b>{}</b></td> <td>{}</td>\n'''.format(_key, _value)
        row += "</tr>\n"
        body += row
    return f"{header}{body}</table>"

class Candidate:
    """
    class for analysing candidate from craco pipeline
    
    note
    ----------
    <1> About each element in the output row
        dtype = np.dtype(
            [
                ('SNR',np.float32), # physical snr
                ('lpix', np.uint16), # l pixel of the detection
                ('mpix', np.uint16), # m pixel of the detection
                ('boxc_width', np.uint8), # boxcar width, 0 for 1 to 7 for 8
                ('time', int), # sample of the detection in a given block
                ('dm', int), # dispersion measure in hardware
                ('iblk', int), # index of the block of the detection, by default each block is 256 samples
                ('rawsn', int), # raw snr
                ('total_sample', int), # sample index over the whole observation
                ('obstime_sec', np.float32), # time difference between the detection and the start of the observation
                ('mjd', np.float64), # detection time in mjd
                ('dm_pccm3', np.float32), # physical dispersion measure
                ('ra_deg', np.float64), 
                ('dec_deg', np.float64)
            ]
        )
        
    <2>  note: mjd value here refers to the final timestamp of the burst 
         i.e., the final boxcar and bottom channel of the burst before de-dispersion
    """
    
    def __init__(
        self, crow, uvsource, calibration_file,
        workdir=None, flagauto=True, extractdata=True,
        flag_ant=None, padding=50, planargs="--ndm 2",
    ):
        """
        initiate the Candidate object with candidate row from the pipeline output
        
        Params
        ----------
        crow: numpy.ndarray
            numpy array loaded from the candidate output
        """
        self.search_output = crow # candidate run output
        self.calibration_file = calibration_file
        self.workdir = workdir
        self.padding = padding
        self.planargs = planargs
        # make work dir if not exists
        if self.workdir is not None:
            if not os.path.exists(workdir): os.makedirs(workdir)
        
        # get basic information from uvsource
        self._fetch_uvprop(uvsource)
        
        # get starting time and ending time of the burst
        self._burstrange()
        
        # TODO:1. allow flagged antenna input based on the candidate file
        # TODO:2. do not load plan automatically.
        if flagauto: # automatically flagging long uvw antennas
            self._flag_ants()
        else:
            self._flag_ants(flag_ant)
            
        # get candidate data from vis...
        # if extractdata:
        self._get_candidate_data(buffer=self.padding)
        self._load_plan()

        self.coord = SkyCoord(
                self.search_output["ra_deg"],
                self.search_output["dec_deg"],
                unit=units.degree,
            )
            
    def _fetch_uvprop(self, uvsource):
        """
        get basic information from uvfits file
        This includes frequencies, time resolution etc.
        """
        if isinstance(uvsource, str):
            uvsource = uvfits.open(uvsource)
        self.uvsource = uvsource
            
        # frequency related
        self.freqs = uvsource.channel_frequencies
        self.fmin = self.freqs[0]; self.fmax = self.freqs[-1]
        self.nchan = self.freqs.shape[0]
        self.foff = (self.fmax - self.fmin) / self.nchan
        # self.foff = np.mean(self.freqs[1:] - self.freqs[:-1])
        
        self.tsamp = uvsource.tsamp.value # in the unit of second
        
    def _burstrange(self):
        """
        get burst range in the unit of sample
        i.e., total sample - boxc_width - t_dis_sample
        """
        dt_dis = calculate_dm_tdelay(
            self.fmin, self.fmax, self.search_output["dm_pccm3"]
        )
        dt_dis_samp = dt_dis // self.tsamp
        _send = self.search_output["total_sample"] # sample end
        _sstart = _send - dt_dis_samp - self.search_output["boxc_width"]
        # _sstart = max(0, _send - dt_dis_samp - self.search_output["boxc_width"])
        
        
        self.burst_range = (int(_sstart), int(_send)) # note this can be out of range...
        logging.info(f"burst spanning from {int(_sstart)} to {int(_send)}...")
        
    def _flag_ants(self, flag_ant=None):
        """
        flagging antennas based on the code we had...
        """
        if flag_ant is None:
            self.uvsource = self.uvsource.set_flagants(
                find_flag(self.uvsource)
            )
        else:
            self.uvsource = self.uvsource.set_flagants(flag_ant)   
        
    def dump_burst_uvfits(self, padding=50, fout="burst.uvfits"):
        """
        dump burst uvfits data based on the burst range...
        """
        # need to figure out how to do this in the new version of uvfits
        pass
        
        # nbl = _vis2nbl(self.uvsource.vis)
        # tt = self.uvsource.vis.size // nbl

        # _sstart, _send = self.burst_range
        
        # ## add padding
        # _sstart = _sstart - padding
        # _send = _send + padding

        # # check if there are within the range
        # if _sstart < 0: _sstart = 0
        # if _send >= tt: _send = tt - 1

        # ### extract tables
        # da_table = self.uvsource.hdulist[0]
        # aux_table = self.uvsource.hdulist[1:]

        # ### only extract _sstart, _send data
        # bu_data = da_table.data[_sstart*nbl:_send*nbl]
        # bu_table = fits.GroupsHDU(bu_data, header=da_table.header)
        # bu_table.header["PZERO4"] = bu_data[0]["DATE"]

        # nhdu = fits.HDUList([bu_table, *aux_table])
        # nhdu.writeto(f"{self.workdir}/{fout}")
    
    def _get_candidate_data(self, buffer=50, uvwave_metrics="mean"):
        """
        get candidate data based on the burst_range
        buffer (padding) is the value for extra integration, by default no extra integration
        """
        logging.info(f"extracting burst data from uvfits file... this can take a long time...")
        _sstart, _send = self.burst_range
        # _sstart = max(0, _sstart - buffer)
        _sstart = _sstart - buffer
        _send = _send + buffer
        
        # self.visrange = (_sstart, _send)
        self.burst_data_dict, self._burst_uvws, _vis_range = self.uvsource.time_block_with_uvw_range((_sstart, _send))
        # self.burst_data_dict = self._visdata_padding(
        #     burst_data_dict, visrange=_visrange, finrange=(_sstart, _send)
        # )
        self.visrange = _vis_range # store the visibility range...
        self.burst_uvw = average_uvws(self._burst_uvws, metrics=uvwave_metrics)
        self.burst_data = bl2array(self.burst_data_dict)
        
    def _load_plan(self):
        """
        load craco plan for calibration... this may be removed later...
        """
        # cause we did autoflagging for uvsource, we don't need to do that for plan again
        if hasattr(self, "visrange"):
            self.burstuvsource = uvfits.open(self.uvsource.filename, skip_blocks=self.visrange[0])
            self.burstplan = craco_plan.PipelinePlan(self.burstuvsource, self.planargs)
            self.plan = self.burstplan
        else:
            self.plan = craco_plan.PipelinePlan(self.uvsource, self.planargs)
            self.burstplan = self.plan


    def _calibrate_data(self, calibration_file):
        """
        calibrate data stored in self.burst_data...
        """
        # self._load_plan() # load craco plan based on the flagged uvsource
        self.calibrator = preprocess.Calibrate(
            plan=self.plan, block_dtype=np.ma.core.MaskedArray,
            miriad_gains_file=calibration_file, 
            baseline_order=self.plan.baseline_order
        )

        # TODO: start calibration
        self.cal_data = self.calibrator.apply_calibration(self.burst_data)

    def _rotate_vis(self):
        """
        rotate the phase center to the candidate coordinate... phase rotate the calibrated data
        """

        lm = coord2lm(self.coord, self.plan.phase_center)
        psvis = pointsource(
            amp=1, lm=lm, freqs=self.plan.freqs, baselines=self.burst_uvw,
            baseline_order=self.plan.baseline_order,
        )

        # we should perform rotating on calibrated data
        if not hasattr(self, "cal_data"): # if no calibrated data created
            self._calibrate_data(self.calibration_file)
        
        self.rotate_data = self.cal_data * np.conj(psvis)[:, :, None, None]

    def _normalise_vis(self, target_input_rms=1, target=True):
        """
        normalise the data by channels (and baselines?)

        by default, we will normalise the phase rotated data (i.e., target is True)
        """
        if target==True:
            self.norm_data = preprocess.normalise(
                self.rotate_data, target_input_rms=target_input_rms
            )

        self.norm_data_pc = preprocess.normalise(
            self.cal_data, target_input_rms=target_input_rms,
        )

    def _load_burst_filterbank(
            self, norm=True, 
            target_input_rms=1,
        ):
        """
        function to load burst filterbank
        """
        if not hasattr(self, "burst_data"):
            self._get_candidate_data(buffer=self.padding)
        
        if not hasattr(self, "rotate_data"):
            self._rotate_vis()

        if not hasattr(self, "norm_data"):
            self._normalise_vis(target_input_rms=target_input_rms)

        if norm:
            self.filterbank = np.nanmean(self.norm_data.real, axis=0) # average on baseline axis
        else:
            self.filterbank = np.nanmean(self.cal_data.real, axis=0)

    def plot_filterbank(self, norm=True, dm=None, keepnan=True):
        """
        plot filterbank with a given dm value...

        Params
        --------
        norm: bool, True by default
            whether use normalised data or original calibrated data
        dm: float or Nonetype, Nonetype by default
            dm value, None for non desperion
        """
        if not hasattr(self, "filterbank"):
            self._load_burst_filterbank(norm=norm, target_input_rms=1)

        fig = plt.figure(figsize=(6, 4))

        filterbank_plot, trange_ = self._dedisperse2tf(dm=dm, norm=norm, keepnan=keepnan)

        grid = mpl.gridspec.GridSpec(
            nrows=5, ncols=5, wspace=0., hspace=0.
        )

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(grid[1:, :-1])

        extent = [
            self.tsamp * trange_[0], # starting time
            self.tsamp * trange_[1], # ending time
            self.freqs[0] / 1e6, self.freqs[-1] / 1e6,
        ]

        ax1.imshow(
            filterbank_plot, 
            aspect="auto", origin="lower", 
            extent=extent,
        )

        ax1.set_xlabel("Time since the observation (s)")
        ax1.set_ylabel("Frequencies (MHz)")

        ax2 = fig.add_subplot(grid[0, :-1], sharex=ax1)
        ### this is something as a function of time
        t = np.linspace(trange_[0], trange_[1], trange_[1]-trange_[0]+1) * self.tsamp
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
        f = np.linspace(self.freqs[0]/1e6, self.freqs[-1]/1e6, self.freqs.shape[0])
        fmin = np.nanmin(filterbank_plot, axis=1)
        fmax = np.nanmax(filterbank_plot, axis=1)
        fmea = np.nanmean(filterbank_plot, axis=1)
        ax3.plot(fmin, f, color="C0")
        ax3.plot(fmax, f, color="C1")
        ax3.plot(fmea, f, color="C2")

        return fig, (ax1, ax2, ax3)

    def _dedisperse2tf(self, dm=None, norm=True, keepnan=True):
        """
        dedisperse the data and return a filterbank
        """
        if not hasattr(self, "filterbank"):
            self._load_burst_filterbank(norm=norm, target_input_rms=1)
        
        if dm is None or dm == 0.0:
            return self.filterbank[:, 0, :].data, self.visrange
        return filterbank_roll(
            tf=self.filterbank[:, 0, :].data, dm=dm,
            freqs=self.freqs, tint=self.tsamp,
            tstart=self.visrange[0], keepnan=keepnan,
        )

    def _dedisperse2ts(self, dm=None, norm=True):
        """
        dedisperse the data and return a timeseries
        i.e., perform averaging (or taking median) along frequency axis
        """
        tf_, trange_ = self._dedisperse2tf(dm=dm, norm=norm)
        return np.nanmedian(tf_, axis=0), trange_

    def _calculate_dmt(self, dmfact=1e2, ndm=100, norm=True):
        """
        work out dmt matrix based on the nominal dm from the search result
        """
        dm_step = self.search_output["dm_pccm3"] / dmfact
        dm_range = self.search_output["dm_pccm3"] + np.arange(-ndm // 2, ndm // 2) * dm_step

        ndm = dm_range.shape[0] # just in case something went wrong...

        trange = (self.visrange[0], 2 * self.visrange[1] - self.visrange[0])
        nt = trange[1] - trange[0] + 1

        dmt = np.zeros((ndm, nt)) * np.nan
        for idm, dm in enumerate(dm_range):
            # get dedispersed dm timeseries and trange...
            tf_ts, trange_ = self._dedisperse2ts(dm=dm, norm=norm)
            range1_, range2_ = _get_overlap_index(trange, trange_)
            
            dmt[idm, range1_[0]:range1_[1]+1] = tf_ts[range2_[0]:range2_[1]+1]
        return dmt, dm_range, trange

    ### plot dm-time plot...
    def plot_dmt(self, dmfact=1e2, ndm=100):
        # calculate dmt matrics
        dmt, dm_range, trange = self._calculate_dmt(dmfact=dmfact, ndm=ndm)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)

        extent = (
            trange[0] * self.tsamp, trange[1] * self.tsamp,
            dm_range[0], dm_range[-1]
        )

        ax.imshow(
            dmt, aspect="auto", extent=extent, 
            interpolation=None, origin="lower",
        )

        ax.scatter(
            self.search_output["obstime_sec"], self.search_output["dm_pccm3"],
            marker="X", s=200, fc="none", ec="black"
        )

        ax.set_xlabel("Time after the observation (s)")
        ax.set_ylabel("Despersion Measure (pc cm^-3)")

        return fig, ax

    ### functions to do with the imaging from here!!!
    def _dedisperse_block(self, norm=True, target_input_rms=1, dm=None):
        """
        function to dedisperse block of data
        """
        # note: we don't need to perform phase rotation here
        if not hasattr(self, "burst_data"):
            self._get_candidate_data(buffer=self.padding)

        if not hasattr(self, "norm_data_pc"):
            self._normalise_vis(target_input_rms=target_input_rms, target=False)

        # use candidate dispersion measure to perform dedispersion
        if dm is None: dm=self.search_output["dm_pccm3"]

        ### initiate the dedisperser
        self.dedisperser = preprocess.Dedisp(
            freqs=self.freqs, tsamp=self.tsamp,
            baseline_order = self.plan.baseline_order, 
            dm_pccc=self.search_output["dm_pccm3"]
        )

        if norm:
            dedisp_data = self.dedisperser.dedisperse(0, self.norm_data_pc)
        else:
            dedisp_data = self.dedisperser.dedisperse(0, self.cal_data)

        nbl, nchan, npol, nt = dedisp_data.shape
        assert npol == 1, f"cannot handle multi polarisations... currently we have {npol}..."

        if nt % 2 != 0: 
            dedisp_data = dedisp_data[..., :-1] # need to make sure nt is even
            self._burst_uvws = {bl: self._burst_uvws[bl][:, :-1] for bl in self._burst_uvws}
        self.dedisp_data = dedisp_data[..., 0, :] # move polarisation axis

    def _image_gridded_data(self, data, imager):
        """
        produce dirty image from gridded data on uv plan
        """
        assert len(data.shape) == 2, "only 2D uv data is allowed..."
        return imager(np.fft.fftshift(data)).astype(np.complex64)

    def _workout_slice_w_center(self, center, length, radius=5):
        """
        work out a slice to index given a radius and the whole length
        # length is used to make sure no indexerror is raised
        """
        return slice(max(center-radius, 0), min(center+radius+1, length))

    def _grid_image_data(self, cutradius=10, ):
        """
        perform gridding and imaging the data
        """
        ### make gridder and imager
        self.gridder = Gridder(self.burstuvsource, self.burstplan, self.burstplan.values) # no values to be parsed
        self.imager = Imager(self.burstuvsource, self.burstplan, self.burstplan.values) # no values to be parsed

        # grid_data with a shape of (npix, npix, nt)
        # here `npix` is the number of gridding on the uv plan
        # `nt` may be not equal to the original `nt`
        # self.grid_data = self.gridder.grid_with_uvws(self.dedisp_data, self._burst_uvws)
        self.grid_data = self.gridder(self.dedisp_data)

        ### TODO: make images each integration, and image for the whole burst
        # make image cube...
        imgcube = np.array([
            self._image_gridded_data(self.grid_data[..., i], self.imager)
            for i in range(self.grid_data.shape[-1])
        ])

        # note from keith: the final image is a complex number,
        # real part and imag part are both images
        # real part stands for timestamp 1, 3, 5, ...
        # imag part stands for timestamp 2, 4, 6, ...
        # TODO: finish this part for self.imgcube

        ### currently, we will use a for loop
        _imgcube = [] # use a list here
        for i in imgcube: #each element is an image (complex number)
            _imgcube.append([i.real])
            _imgcube.append([i.imag])
        self.imgcube = np.concatenate(_imgcube)

        ### perform zoom in
        nt, llen, mlen = self.imgcube.shape
        lpix = self.search_output["lpix"]
        mpix = self.search_output["mpix"]
        self.imgzoomcube = self.imgcube[
            :, self._workout_slice_w_center(mpix, mlen, cutradius),
            self._workout_slice_w_center(lpix, llen, cutradius),
        ]

        ### work out image stats
        self.imgmax = np.nanmax(self.imgzoomcube, axis=(1, 2))
        self.imgstd = np.nanstd(self.imgcube, axis=(1, 2))
        self.imgsnr = self.imgmax / self.imgstd
        
    def _plot_single_image_wo_wcs(self, imgdata, vmin, vmax):
        """
        plot a single image without wcs, this is mainly used for zoom in image
        """
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(
            imgdata, origin="lower",
            vmin=vmin, vmax=vmax,
            aspect="auto"
        )
        
        return fig, ax

    def _make_burst_images(
            self, gif=True, vmin=0, vmax=50
    ):
        """
        make burst image, zoom in version for the detected burst...
        """
        # if no imageio found... set gif to false...
        try: import imageio
        except: gif = False

        imglist = []
        nt, llen, mlen = self.imgcube.shape

        for it in range(nt):
            # img_zoom = self.imgcube[it][
            #     self._workout_slice_w_center(lpix, llen, cutradius),
            #     self._workout_slice_w_center(mpix, mlen, cutradius),
            # ]
            img_zoom = self.imgzoomcube[it]

            fig, _ = self._plot_single_image_wo_wcs(
                imgdata=img_zoom.real, vmin=vmin, vmax=vmax,
            )

            imglist.append(importfig2img(fig))

        self.imglist = imglist # this store burst images...

        if gif == True: # make gif images...
            imageio.mimwrite(f"{self.workdir}/burst.gif", self.imglist, duration=20, loop=0)
        else:
            zoomdir = f"{self.workdir}/burstzoom/"
            if not os.path.exists(zoomdir):
                os.makedirs(zoomdir)
            for iimg, img in enumerate(self.imglist):
                img.save(f"{zoomdir}/{iimg}.png")

    def _make_field_image(
        self, vmin=None, vmax=None, wwcs=True, save=True
    ):
        # this will make the image of the whole field
        # we will take the median of the whole image stack
        # this equal to the incoherent sum in image domain
        medimg = np.nanmedian(self.imgcube.real, axis=0)
        stdimg = np.nanstd(self.imgcube.real, axis=0)

        fig = plt.figure(figsize=(12, 4))

        projection=self.plan.wcs if wwcs else None
        #maximum image
        ax = fig.add_subplot(1, 3, 1, projection=projection)

        maxidx = np.argmax(self.imgsnr)
        
        ax.imshow(
            self.imgcube[maxidx], vmin=vmin, vmax=vmax, 
            origin="lower", aspect="auto",
        )
        ax.set_title("max image")

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

        if save:
            fig.savefig(f"{self.workdir}/burst_field_image.jpg", bbox_inches="tight")
            plt.close()
        else:
            return fig

    def _make_detection_images(self, vmin=0, vmax=50):
        """
        make a series of images based on the detection sample
        """
        dets = self.search_output["total_sample"]
        viss = self.visrange[0]

        # TODO: check if this make sense
        imgidx_e = dets - viss
        imgidx_s = dets - viss - self.search_output["boxc_width"]


        fig = plt.figure(figsize=(24, 3))
        for i, iimg in enumerate(range(imgidx_s, imgidx_e + 1)):
            if i >= 8: break
            ax = fig.add_subplot(1, 8, i+1)
            ax.imshow(
                self.imgzoomcube[iimg].real, 
                vmin=vmin, vmax=vmax, aspect="auto",
                origin="lower",
            )

            ax.set_title(f"{iimg} IMG_SNR={self.imgsnr[iimg]:.2f}")
        
        fig.savefig(f"{self.workdir}/burst_snapshots.jpg", )
        plt.close()

    def _plot_image_stats(self):
        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(3, 1, 1)
        ax.plot(self.imgsnr, label="snr")
        ax.legend()

        ax = fig.add_subplot(3, 1, 2)
        ax.plot(self.imgstd, label="std")
        ax.legend()

        ax = fig.add_subplot(3, 1, 3)
        ax.plot(self.imgmax, label="max pixel")
        ax.legend()
        

        fig.savefig(f"{self.workdir}/image_stats.jpg", bbox_inches="tight")
        plt.close()
        

    ###### main public functions here!
    def run_filterbank(
        self, norm=True, target_input_rms=1,
        dmfact=1e2, ndm=2e2,
    ):
        """
        functions to extract data and make filterbank data
        and then plot diagnostic plots based on filterbank data

        Params # TODO
        ----------
        norm...
        """
        # load data, calibrate data, rotate data, normalise data, make filterbank
        if not hasattr(self, "burst_data"):
            self._get_candidate_data(buffer=self.padding)
        if not hasattr(self, "cal_data"):
            self._calibrate_data(self.calibration_file)
        if not hasattr(self, "rotate_data"):
            self._rotate_vis()
        if not hasattr(self, "norm_data"):
            self._normalise_vis(target_input_rms=target_input_rms)
        if not hasattr(self, "filterbank"):
            self._load_burst_filterbank(norm=norm, target_input_rms=1)

        ### do some plots...
        # 0-DM dedispered data
        fig, ax = self.plot_filterbank(dm=0.)
        fig.suptitle("Dedispersed at DM=0")
        fig.savefig(f"{self.workdir}/filterbank_dm0.0.jpg", bbox_inches="tight")
        plt.close()

        # nominal-DM dedispersed data
        fig, ax = self.plot_filterbank(dm=self.search_output["dm_pccm3"])
        fig.suptitle("Dedispersed at DM={:.2f}".format(self.search_output["dm_pccm3"]))
        fig.savefig(f"{self.workdir}/filterbank_dm{self.search_output['dm_pccm3']:.1f}.jpg", bbox_inches="tight")
        plt.close()

        fig, ax = self.plot_filterbank(dm=self.search_output["dm_pccm3"], keepnan=False)
        fig.suptitle("Dedispersed at DM={:.2f}".format(self.search_output["dm_pccm3"]))
        fig.savefig(f"{self.workdir}/filterbank_dm{self.search_output['dm_pccm3']:.1f}_center.jpg", bbox_inches="tight")
        plt.close()

        # plot DM-t plot, butterfly!!!
        fig, ax = self.plot_dmt(dmfact=dmfact, ndm=ndm)
        fig.savefig(f"{self.workdir}/butterfly_plot.jpg", bbox_inches="tight")
        plt.close()

    def run_imager(
        self, norm=True, target_input_rms=1, cutradius=10, 
        vmin=0., vmax=50., gif=True, wwcs=True
    ):
        """
        function to perform dedispersion on visibility data
        make cutout images for the burst and plot diagnostic plots
        """
        #load data, calibrate data, normalise data, dedisperse data
        if not hasattr(self, "burst_data"):
            self._get_candidate_data(buffer=self.padding)
        if not hasattr(self, "cal_data"):
            self._calibrate_data(self.calibration_file)
        if not hasattr(self, "norm_data_pc"):
            self._normalise_vis(target_input_rms=target_input_rms, target=False)
        if not hasattr(self, "dedisp_data"):
            self._dedisperse_block(
                norm=norm, target_input_rms=target_input_rms,
                dm=self.search_output["dm_pccm3"],
            )
            # TODO: need to double check if the data has already
            # been dedispered before, and if it is dedispersed with 
            # the correct dm
        
        # make gridding and perform imaging...
        self._grid_image_data(cutradius=cutradius)

        ### plots
        # plot burst image
        self._make_burst_images(
            gif=gif, vmin=vmin, vmax=vmax,
        )
        # make image of the field
        self._make_field_image(wwcs=wwcs, vmin=vmin, vmax=vmax,)
        self._make_detection_images(vmin=vmin, vmax=vmax,)
        self._plot_image_stats()

    def _make_srcinfo(self):
        """
        create a list contains source/burst information
        """
        ### information from uvsource
        # make information for coordinate
        pccoord = self.uvsource.get_target_skycoord()
        self.fitsinfo=[
            ["Target", self.uvsource.target_name],
            ["Center Coord",  pccoord.to_string("hmsdms")],
            ["Telescope", "ASKAP CRACO"],
            ["Tsamp", "{:.2f} ms".format(self.tsamp * 1e3)],
            ["Freq", "{:.1f} - {:.1f} MHz".format(self.freqs[0]/1e6, self.freqs[-1]/1e6)],
            ["Fctr", "{:.1f} MHz".format((self.freqs[0] + self.freqs[-1]) / 2 / 1e6)],
            ["Nchan", "{}".format(self.nchan)],
        ]

        ### information from candidates
        # work out the mjd time of the burst at inf freq
        _mjd = self.search_output["mjd"]
        _delay = calculate_dm_tdelay(self.freqs.min(), np.inf, self.search_output["dm_pccm3"])
        _day = 24 * 3600
        _boxcwidth = self.search_output["boxc_width"] + 1
        _mjd = _mjd - _delay / _day - _boxcwidth * self.tsamp / _day

        # check if coord exists
        # if not hasattr(self, "coord"):
        #     self.coord = SkyCoord(
        #         self.search_output["ra_deg"],
        #         self.search_output["dec_deg"],
        #         unit=units.degree,
        #     )

        self.srcinfo=[
            ["Coord", self.coord.to_string("hmsdms")],
            ["Gal Coord", self.coord.galactic.to_string("decimal")],
            ["SNR", "{:.2f}".format(self.search_output["SNR"])],
            ["DM", "{:.2f} pc cm^-3".format(self.search_output["dm_pccm3"])],
            ["Width", "boxcar {} => {:.1f} ms".format(_boxcwidth, _boxcwidth * self.tsamp * 1e3)],
            ["TimeSec", "{:.1f} s".format(self.search_output["obstime_sec"])],
            ["MJDinf", "{:.9f}".format(_mjd)], 
        ]

    def create_webpage(self):
        """
        function to create a webpage to store all information
        """
        self._make_srcinfo()

        head = """<head>\n{}</head>\n""".format(_webpage_style())

        ### information
        info = """<table style="width: 95%; margin-left: auto; margin-right: auto;">\n"""
        info += """<td style="width: 80%;">\n"""
        info += _webpage_info_table(self.fitsinfo)
        # info += """</tr>\n<tr>\n"""
        info += _webpage_info_table(self.srcinfo)
        info += """</td>\n"""
        info += """<td style="width: 20%;"><img src="burst.gif" style="width: 100%;"></td>\n"""
        info += """</tr>\n</table>\n"""

        ### filterbanks
        burstplots = """<table class="fourcol" style="margin-left: auto; margin-right: auto;">
<tr>
    <td><img src="filterbank_dm0.0.jpg" style="width: 100%;"></td>
    <td><img src="filterbank_dm{:.1f}.jpg" style="width: 100%;"></td>
    <td><img src="filterbank_dm{:.1f}_center.jpg" style="width: 100%;"></td>
    <td><img src="butterfly_plot.jpg" style="width: 100%;"></td>
</tr>
</table>
""".format(self.search_output["dm_pccm3"], self.search_output["dm_pccm3"])

        burstimages = """<table style="width: 95%; margin-left: auto; margin-right: auto;">
<tr>
    <td><img src="burst_snapshots.jpg" style="width: 100%;"></td>
</tr>
</table>
"""

        fieldplots = """<table style="width: 80%; margin-left: auto; margin-right: auto;">
<tr>
    <td style="width: 60%"><img src="burst_field_image.jpg" style="width: 100%;"></td>
    <td style="width: 20%"><img src="image_stats.jpg" style="width: 100%;"></td>
</tr>
</table>
"""
        html =  f"<html>{head}{info}{burstplots}{burstimages}{fieldplots}</html>"

        with open(f"{self.workdir}/burst.html", "w") as fp:
            fp.write(html)

        #save the html file as the png file
        # make sure you install weasyprint v52.5
        self._export_html2png()

    def _export_html2png(self):
        """
        export saved webpage to png files...
        """
        try: from weasyprint import HTML, CSS 
        except ImportError: 
            logging.info("no weasyprint package found... will not convert html to png")
            return

        ### start converting
        html = HTML(f"{self.workdir}/burst.html")
        html.write_png(
            f"{self.workdir}/burst.html.png",
            presentational_hints=True,
            stylesheets=[CSS(string='''@page {
                size: A3 landscape; 
                margin: 0mm; 
                background-color: white;
            }''')],
        )


def load_cands(fname, maxcount=None):
    dtype = np.dtype(
        [
            ('SNR',np.float32), # physical snr
            ('lpix', np.uint16), # l pixel of the detection
            ('mpix', np.uint16), # m pixel of the detection
            ('boxc_width', np.uint8), # boxcar width, 0 for 1 to 7 for 8
            ('time', int), # sample of the detection in a given block
            ('dm', int), # dispersion measure in hardware
            ('iblk', int), # index of the block of the detection, by default each block is 256 samples
            ('rawsn', int), # raw snr
            ('total_sample', int), # sample index over the whole observation
            ('obstime_sec', np.float32), # time difference between the detection and the start of the observation
            ('mjd', np.float64), # detection time in mjd
            ('dm_pccm3', np.float32), # physical dispersion measure
            ('ra_deg', np.float64), 
            ('dec_deg', np.float64)
        ]
    )
    c = np.loadtxt(fname, dtype=dtype, max_rows=maxcount)
    return c


# use it in CLU
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Produce Candidate Plots', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-uv", "--uv", type=str, help="Path to the uvfits file", default=None)
    parser.add_argument('-cal','--calibration',  type=str, help="Path to the calibration file", default=None)
    parser.add_argument("-can", "--candidate", type=str, help="Path to the candidate file", default=None)
    parser.add_argument("-idx", "--index", type=int, help="candidate index in the candidate file", default=0)
    parser.add_argument("-norm", action='store_true', help="Normalise the data (baseline subtraction and rms setting to 1)",default = True)
    parser.add_argument("--target_input_rms", type=int, default=1)
    parser.add_argument("--remove_aux", type=bool, help="wheter to clean uvdata created by cracoplan function", default=True)
    parser.add_argument("--fout", type=str, help="burst uvfits file filename", default="burst.uvfits")
    parser.add_argument("-pad", "--padding", type=int, help="padding when extracting data", default=50,)
    parser.add_argument("-p", "--path", type=str, help="place to save all images, html files", default="./")
    parser.add_argument("--vmin", type=float, help="minimum value for imshow", default=0.)
    parser.add_argument("--vmax", type=float, help="maximum value for imshow", default=50.)
    parser.add_argument("--cutradius", type=int, help="zoom in radius for the cutout", default=10)
    
    values = parser.parse_args()

    c = load_cands(values.candidate)
    cand = Candidate(
        crow=c[values.index], uvsource=values.uv,
        calibration_file=values.calibration, 
        extractdata=False, workdir=values.path,
        padding=values.padding,  loadplan=True
    )

    cand.dump_burst_uvfits(fout=values.fout)
    cand.run_filterbank(norm=values.norm, target_input_rms=values.target_input_rms)
    cand.run_imager(
        norm=values.norm, target_input_rms=values.target_input_rms,
        vmin=values.vmin, vmax=values.vmax, cutradius=values.cutradius,
    )
    cand.create_webpage()

    if values.remove_aux:
        os.system("rm $PWD/uv_data.*.txt")

if __name__ == "__main__":
    _main()

