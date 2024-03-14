
import logging
import numpy as np
from craco.preprocess import fast_preprpocess
from craco.vis_subtractor import VisSubtractor

from craft import uvfits, craco_plan
from craco import uvfits_meta, calibration

log = logging.getLogger(__name__)

def original_calibrate_input(solarray, values_subtract, values_target_input_rms, input_flat_raw):
    '''
    Apply calibration solutions -  Multiply by solution aray
        Need to make a copy, as masked arrays loose the mask if you *= with an unmasked array
    '''
    log.info("Starting calibration")
    if solarray is not None:
        # If data is already polsummed, then average the solutions before multiplying
        if solarray.ndim == 4 and input_flat_raw.ndim == 3:
            sols = solarray.mean(axis=2)
            input_flat = sols*input_flat_raw
        else:
            input_flat = solarray*input_flat_raw
    else:
        input_flat = input_flat_raw.copy()

    log.info("Starting normalisation")

    # subtract average over time
    if values_subtract >= 0:
        subtractor = VisSubtractor(input_flat.shape, values_subtract)
        input_flat = subtractor(input_flat)

    log.info("Starting pol averaging")
    # average polarisations, if necessary
    if input_flat.ndim == 4:
        npol = input_flat.shape[2]
        assert npol == 1 or npol == 2, f'Invalid number of polarisations {npol} {input_flat.shape}'
        if npol == 1:
            input_flat = input_flat[:,:,0,:]
        else:
            input_flat = input_flat.mean(axis=2)

    log.info("Starting rms computation")
    # scale to give target RMS
    targrms = values_target_input_rms
    if  targrms > 0:
        # calculate RMS
        real_std = input_flat.real.std()
        imag_std = input_flat.imag.std()
        input_std = np.sqrt(real_std**2+ imag_std**2)/np.sqrt(2) # I think this is right do do quadrature noise over all calibrated data
        # noise over everything
        stdgain = targrms / input_std
        log.info('Input RMS (real/imag) = (%s/%s) quadsum=%s stdgain=%s targetrms=%s', real_std, imag_std, input_std, stdgain, targrms)
        log.info("Applying the rms now")
        input_flat *= stdgain

    return input_flat

def original_apply_cal(solarray, input_flat_raw):
    log.info("Starting calibration")
    if solarray is not None:
        # If data is already polsummed, then average the solutions before multiplying
        if solarray.ndim == 4 and input_flat_raw.ndim == 3:
            sols = solarray.mean(axis=2)
            input_flat = sols*input_flat_raw
        else:
            input_flat = solarray*input_flat_raw
    else:
        input_flat = input_flat_raw.copy()
    
    return input_flat

def original_subtract_vis(values_subtract, input_flat):
    # subtract average over time
    if values_subtract >= 0:
        subtractor = VisSubtractor(input_flat.shape, values_subtract)
        input_flat = subtractor(input_flat)
    
    return input_flat


def original_apply_rms(values_target_input_rms, input_flat):
    log.info("Starting rms computation")
    # scale to give target RMS
    targrms = values_target_input_rms
    if  targrms > 0:
        # calculate RMS
        real_std = input_flat.real.std()
        imag_std = input_flat.imag.std()
        input_std = np.sqrt(real_std**2+ imag_std**2)/np.sqrt(2) # I think this is right do do quadrature noise over all calibrated data
        # noise over everything
        stdgain = targrms / input_std
        log.info('Input RMS (real/imag) = (%s/%s) quadsum=%s stdgain=%s targetrms=%s', real_std, imag_std, input_std, stdgain, targrms)
        log.info("Applying the rms now")
        input_flat *= stdgain

    return input_flat


fname = "/data/craco/gup037/DATA/SB057841/DATA/DATA_01/craco/SB057841/scans/00/20240121205332/b00.uvfits"
meta_name = "/data/craco/gup037/DATA/SB057841/SB057841/SB57841.json.gz"
calname = "/data/craco/gup037/DATA/SB057841/SB057841/cal/00/b00.aver.4pol.smooth.npy"


values = craco_plan.get_parser().parse_args(["--flag-ants", "12,15,20,30", "--calibration", calname])
f = uvfits_meta.open(fname, metadata_file = meta_name)
f.set_flagants(values.flag_ants)

plan = craco_plan.PipelinePlan(f, values)
calsoln = calibration.CalibrationSolution(plan)

block0, uvws0 = next(f.fast_time_blocks(nt = 256))
block0 = block0.squeeze()


#variables needed for fast_preprocess
input_block = block0.copy()
input_data = input_block.data
input_mask = input_block.mask

nbl, nf, nt = input_block.shape
isubblock = 0
output_buf = np.zeros_like(input_data)
#output_mask = np.zeros_like(input_mask)

Ai = np.zeros((nbl, nf), dtype=np.complex64)
Qi = np.zeros((nbl, nf), dtype=np.complex64)
N = np.ones((nbl, nf), dtype=np.int16)

cas = np.zeros((nf, nt), dtype=np.float64)
crs = np.zeros((nf, nt), dtype=np.float64)
cas_N = np.zeros((nf, nt), dtype=np.int16)

cal= calsoln.solarray.mean(axis=2).squeeze()
calsoln_data = cal.data
calsoln_mask = cal.mask

def test_calibration_equality():
    original_calibrated_data = original_apply_cal(calsoln.solarray, block0)
    fast_calibrated_data = fast_preprpocess(input_block, input_mask, output_buf, isubblock, Ai, Qi, N, calsoln_data, calsoln_mask, cas, crs, cas_N, target_input_rms=None, sky_sub=False, reset_scales=True)

    assert np.allclose(original_calibrated_data, fast_calibrated_data)
    
