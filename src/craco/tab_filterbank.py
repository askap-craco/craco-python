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


def run(f, values):
    # we assume we are not using simulated data...
    block_dtype = np.ma.core.MaskedArray

    uvsource = uvfits.open(values.uv)
    ## check if this need to change...
    # plan = craco_plan.PipelinePlan(uvsource, values)
    plan_arg = f"--ndm {values.ndm} --max-nbl {values.max_nbl} --flag-ant {values.flag_ant}"
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
        'src_raj':coord.ra.deg,
        'src_dej':coord.dec.deg,
        'tstart':plan.tstart.utc.mjd,
        'tsamp':plan.tsamp_s.value, # in the unit of second
        'fch1':plan.fmin,
        'foff':plan.foff,
        'source_name':vis_source.target_name,
    }

    foutname = f+'.fil'
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
    for blk, bluvw in uvsource.time_blocks_with_uvws(values.nt):

        iblk += 1
        # logger.info(f"processing block {iblk}...")
        print(f"processing block {iblk}...")

        # take the mean of uvw here to represent the block
        bluvw_ave = average_uvw(bluvw, metrics="mean")
        
        # make a model for a unit Jansky source
        psvis = pointsource(1, lm, plan.freqs, plan.baseline_order, bluvw_ave)

        # perform calibration
        blk = bl2array(blk)
        cal_vis = calibrator.apply_calibration(blk)
        # phase rotation
        # this need to be done for each sample...
        rot_vis = cal_vis * np.conj(psvis)[:, :, None, None]
        # # flagging
        # flag_vis, _, _, _ = rfi_cleaner.run_IQRM_cleaning(
        #     np.abs(rot_vis), False, False, False, False, True, True
        # )
        # # normalisation
        # fin_vis = preprocess.normalise(
        #     flag_vis, target_input_rms=values.target_input_rms
        # )

        fin_vis = preprocess.normalise(
            rot_vis, target_input_rms=values.target_input_rms
        )

        out_tf = fin_vis.real.mean(axis=0)
        assert out_tf.shape[1] == 1, "not single polarsiation found..."
        out_tf = out_tf[:, 0, :].T.data.astype(np.float32) # shape: frequency vs time

        # mask = np.ma.getmask(out_tf)
        # data = np.ma.getdata(out_tf)
        # data[mask] = np.nan

        out_tf.tofile(fout.fin)

    fout.fin.close()
    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Produces a tied array beam filterbank from a UVFITS file', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-uv", "--uv", type=str, help="Path to the uvfits file", default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-c','--calibration',  type=str, help="Path to the calibration file", default=None)
    parser.add_argument("-t", "--target", type=str, help="coordinate of the phase center", default="pc")
    parser.add_argument("-nt", "--nt", type=int, help="number of block reduce in each iteration...", default=128)
    parser.add_argument("--target_input_rms", type=int, default=1)
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