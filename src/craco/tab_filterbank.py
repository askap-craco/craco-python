#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craft import uvfits
from craft import sigproc
from craft import craco_plan
from craft.craco import pointsource, coord2lm, bl2array
from craco import preprocess

from craft.cmdline import strrange

from astropy.coordinates import SkyCoord

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def parse_target(target, phase_center):
    '''
    Returns  SkyCoord given the target string
    if target is 'pc' returns the phase_center
    if target starts with 'offset:', it assumes comma separated angles parseable as 
    astropy.Angle e.g. 'offset:0.1d,0.2d'
    else parses as a SkyCoord directrly
    '''
    if target == 'pc':
        coord = phase_center
    elif target.startswith('offset:'):
        offset_str = values.target[length('offset:'):].split(',')
        psi,theta = map(Angle, offset_str)
        coord = craco.psitheta2coord((psi,theta), phase_center)
    else:
        coord = SkyCoord(target)

    return coord

def average_uvw(bluvws, metrics="mean"):
    """
    Returns a new dictionary for the baseline uvw values from a list of samples...
    Currently only support mean value for all possible uvw
    """

    assert metrics in ["mean", "start", "end"], "we support `mean`, `start`, and `end`"

    if metrics == "mean":
        return {blid: bluvws[blid].mean(axis=-1) for blid in bluvws}
    if metrics == "start":
        return {blid: bluvws[blid][:, 0] for blid in bluvws}
    if metrics == "end":
        return {blid: bluvws[blid][:, -1] for blid in bluvws}

def phase_rotate(blk, model):
    """
    there is broadcasting issue here,
    there might be a clever way in numpy.
    Here we just use for loop...
    """
    d_nbl, d_nchan, d_npol, d_nt = blk.shape
    m_nbl, m_nchan = model.shape
    assert d_nbl == m_nbl, "not the same number of baselines..."
    assert d_nchan == m_nchan, "not the same number of channels..."

    blk_rephase = blk.copy()

    for ipol in range(d_npol):
        for it in range(d_nt):
            blk_rephase[:, :, ipol, it] = blk[:, :, ipol, it] * np.conj(model)
    return blk_rephase

def nbl2nant(nbl):
    return int((1 + np.sqrt(1 + 8*nbl)) / 2)

def get_ant_idx(nbl, flag_ant=None):
    if flag_ant is None: return np.ones(nbl, dtype=bool)
    if isinstance(flag_ant, str):
        nant = nbl2nant(nbl)
        blant = np.array([
            (i, j) for i in range(nant) for j in range(i+1, nant)
        ])
        flag_ant = np.array(strrange(flag_ant)) - 1 # make it zero-indexed...
        return (~np.isin(blant[:, 0], flag_ant)) & (~np.isin(blant[:, 1], flag_ant))


def open_after_seeking(fname, seek_sec = None, seek_samps = None):
    tmp = uvfits.open(fname)
    nsamps_total = int(tmp.vis.size // tmp.nbl)

    if seek_sec is not None:
        tsamp = tmp.tsamp.value
        seek_samps = int(np.round(seek_sec / tsamp))
    
    assert seek_samps < nsamps_total, "Requested seek_samps ({seek_samps}) is bigger than total_nsamps ({nsamps_total})"

    tmp.close()

    f = uvfits.open(fname, skip_blocks=seek_samps)

    return f


def run(f, values):
    # we assume we are not using simulated data...
    block_dtype = np.ma.core.MaskedArray
    

    uvsource = open_after_seeking(values.uv, seek_samps = values.seek_samps)
    nsamps_total = int(uvsource.vis.size // uvsource.nbl)
    nsamps_to_process = values.process_samps

    if nsamps_to_process < 0 or nsamps_to_process > nsamps_total:
        nsamps_to_process = nsamps_total

    ## check if this need to change...
    # plan = craco_plan.PipelinePlan(uvsource, values)
    plan_arg = f"--ndm {values.ndm} --max-nbl {values.max_nbl}" # --flag-ant {values.flag_ant}"
    if not (values.flag_ant is None): plan_arg += f" --flag-ant {values.flag_ant}"

    plan = craco_plan.PipelinePlan(uvsource, plan_arg) # --flag-ant 13-15,29")
    calibrator = preprocess.Calibrate(
        plan = plan, block_dtype=block_dtype, 
        miriad_gains_file=values.calibration, 
        baseline_order=plan.baseline_order,
    )
    
    #coord = SkyCoord('13h13m12s+12d12m12s')
    coord = parse_target(values.target, plan.phase_center)

    lm = coord2lm(coord, plan.phase_center)
    npol = 1
    vis_source = uvsource

    hdr = {
        'nbits':32,
        'nchans':plan.freqs.shape[0],
        'nifs':npol, 
        'src_raj_deg':coord.ra.deg,
        'src_dej_deg':coord.dec.deg,
        'tstart':plan.tstart.utc.mjd,
        'tsamp':plan.tsamp_s.value, # in the unit of second
        'fch1':plan.fmin / 1e6,
        'foff':plan.foff / 1e6,
        'source_name':vis_source.target_name,
    }

    foutname = f+'.fil' if not f.endswith(".fil") else f
    fout = sigproc.SigprocFile(foutname, 'wb', hdr)

    calibrator = preprocess.Calibrate(
        plan = plan, block_dtype=block_dtype, 
        miriad_gains_file=values.calibration, 
        baseline_order=plan.baseline_order
    )

    rfi_cleaner = preprocess.RFI_cleaner(
        block_dtype=block_dtype, 
        baseline_order=plan.baseline_order
    )

    iblk = 0
    nblocks_to_process = int(nsamps_to_process / values.nt)
    for blk, bluvw in uvsource.time_blocks_with_uvws(values.nt):
        try:
            iblk += 1
            # logger.info(f"processing block {iblk}...")
            print(f"processing block {iblk}...")

            # take the mean of uvw here to represent the block
            bluvw_ave = average_uvw(bluvw, metrics="mean")
            
            # make a model for a unit Jansky source
            psvis = pointsource(1, lm, plan.freqs, plan.baseline_order, bluvw_ave)

            # perform calibration
            blk = bl2array(blk)
            _vis = calibrator.apply_calibration(blk)
            # phase rotation
            # this need to be done for each sample...
            _vis = _vis * np.conj(psvis)[:, :, None, None]
            # # flagging
            # flag_vis, _, _, _ = rfi_cleaner.run_IQRM_cleaning(
            #     np.abs(rot_vis), False, False, False, False, True, True
            # )

            if values.norm:
                _vis = preprocess.normalise(
                    _vis, target_input_rms=values.target_input_rms
                )
            # shape: nbl, nchan, npol, nt

            out_tf = _vis.real.mean(axis=0)
            # ### if using weighted_snr...
            # if values.weighted_snr is None:
            #     out_tf = fin_vis.real.mean(axis=0)
            # else:
            #     ### shape: nbl, nchan, npol
            #     bp_snr = np.load(values.weighted_snr)[..., 0] # just pick up one polarisation atm...
            #     ### work out flagged antenna...
            #     _nbl, _nchan = bp_snr.shape
            #     ant_idx = get_ant_idx(_nbl, values.flag_ant)
            #     bp_snr = bp_snr[ant_idx, ...]
            #     #####
            #     bp_var = 1 / bp_snr ** 2
            #     bp_var = bp_var[..., None, None] # make dimension match...
            #     bp_weight = bp_var / np.nansum(bp_var, axis=0, keepdims=True) # sum over baseline

            #     weight_tf = fin_vis.real * bp_weight
            #     out_tf = fin_vis.sum(axis=0)
                
            assert out_tf.shape[1] == 1, "not single polarsiation found..."
            out_tf = out_tf[:, 0, :].T.data.astype(np.float32) # shape: frequency vs time

            # mask = np.ma.getmask(out_tf)
            # data = np.ma.getdata(out_tf)
            # data[mask] = np.nan

            out_tf.tofile(fout.fin)

            if iblk >= nblocks_to_process:
                raise KeyboardInterrupt

        except KeyboardInterrupt:
            fout.fin.close()
            os.system("rm uv_data*.txt")
            break

    fout.fin.close()
    os.system("rm uv_data*.txt")
    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Produces a tied array beam filterbank from a UVFITS file', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-uv", "--uv", type=str, help="Path to the uvfits file", default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-c','--calibration',  type=str, help="Path to the calibration file", default=None)
    parser.add_argument("-t", "--target", type=str, help="coordinate of the phase center", default="pc")
    parser.add_argument("-nt", "--nt", type=int, help="number of block reduce in each iteration...", default=128)
    # parser.add_argument("-w", "--weighted-snr", type=str, help="snr weighted bandpass...", default=None)
    parser.add_argument("-norm", action='store_true', help="Normalise the data (baseline subtraction and rms setting to 1)",default = False)
    parser.add_argument("--target_input_rms", type=int, default=1)
    parser.add_argument("--seek_samps", type=int, help="Seek x samps in to the file", default=0)
    parser.add_argument("--process_samps", type=int, help="Process only x samples in the file (say -1 to process until the end of the file)",  default=-1)
    parser.add_argument("--ndm", type=int, default=2)
    parser.add_argument("--max-nbl", type=int, default=465)
    parser.add_argument("--flag-ant", type=str, )

    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    for f in values.files:
        run(f, values)
    

if __name__ == '__main__':
    _main()
