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

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def parse_target(target:str, phase_center:SkyCoord):
    '''
    Returns  SkyCoord given the target string
    if target is 'pc' returns the phase_center
    if target starts with 'offset:', it assumes comma separated angles parseable as 
    astropy.Angle e.g. 'offset:0.1d,0.2d'
    else parses as a SkyCoord directrly
    '''
    if target == 'pc':
        coord = phase_center
    if target.startswith('offset:'):
        offset_str = values.target[length('offset:'):].split(',')
        psi,theta = map(Angle, offset_str)
        coord = craco.psitheta2coord((psi,theta), phase_center)
    else:
        coord = SkyCoord(target)

    return coord


def run(f, values):
    uvsource = uvfits.open(values.uv)
    plan = craco_plan.PipelinePlan(uvsource, values)
    calibrator = preprocess.Calibrate(plan = plan, block_dtype=block_type, miriad_gains_file=args.calfile, baseline_order=plan.baseline_order)
    
    #coord = SkyCoord('13h13m12s+12d12m12s')
    coord = parse_target(values.target, plan.phase_center)

    lm = craft.coord2lm(coord, plan.phase_center)
    npol = 1
    vis_source = uvsource
    hdr = {'nbits':32,
           'nchans':vis_source.nchan,
           'nifs':npol, 
           'src_raj':pos.ra.deg,
           'src_dej':pos.dec.deg,
           'tstart':vis_source.tstart.utc.mjd,
           'tsamp':vis_source.inttime,
           'fch1':vis_source.fch1,
           'foff':vis_source.foff,
           'source_name':vis_source.target
    }

    foutname = f+'.fil'
    fout = sigproc.SigprocFile(foutname, 'wb', hdr)

    for blk in uvsource.time_blocks(values.nt):
        # UVWS changes every bloc. Update pointsource
        
        psvis = craco.pointsource(1, lm, plan.freqs, plan.baseline_order,blk)
        cal_vis = calibrator.calibrate(blk)
        out_vis = cal_data * np.conj(psvis)
        out_tf = out_vis.real.sum(axis=0) # ??? TODO: check axis
        out_tf.writeto(fout.fin)

    fout.close()
    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Produces a tied array beam filterbank from a UVFITS file', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-c','--calibration'
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    for f in files:
        run(f, values)
    

if __name__ == '__main__':
    _main()
