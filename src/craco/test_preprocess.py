
import logging
import numpy as np
from craco.preprocess import fast_preprocess, fast_preprocess_single_norm
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
global_input_data = input_block.data
input_mask = input_block.mask

nbl, nf, nt = input_block.shape
isubblock = 0
global_output_buf = np.zeros_like(global_input_data)
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
    fast_calibrated_data = fast_preprocess(input_block, input_mask, global_output_buf, isubblock, Ai, Qi, N, calsoln_data, calsoln_mask, cas, crs, cas_N, target_input_rms=None, sky_sub=False, reset_scales=True)

    assert np.allclose(original_calibrated_data, fast_calibrated_data)
    
def test_fast_preprocess_single_norm_with_zero():
    input_data = np.zeros_like(global_input_data, dtype=np.complex64)
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    Ai = np.zeros(1, dtype=np.complex64)
    Qi = np.zeros(2, dtype=np.float64)
    N = np.ones(1, dtype=np.int32) 
    calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    fast_preprocess_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms, sky_sub)
    assert np.isclose(Ai.real, 0)
    assert np.isclose(Ai.imag, 0)
    assert np.isclose(Qi[0], 0)
    assert np.isclose(Qi[0], 0)
    assert N == input_data.size + 1
    assert np.all(output_buf.real == 0)
    assert np.all(output_buf.imag == 0)


def test_fast_preprocess_single_norm_with_ones():
    input_data = np.zeros_like(global_input_data, dtype=np.complex64) + (1+1j)
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    Ai = np.zeros(1, dtype=np.complex64)
    Qi = np.zeros(2, dtype=np.float64)
    N = np.ones(1, dtype=np.int32) 
    calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    fast_preprocess_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms, sky_sub)
    assert np.isclose(Ai.real, 1)
    assert np.isclose(Ai.imag, 1)
    assert np.isclose(Qi[0], 0)
    assert np.isclose(Qi[0], 0)
    assert N == input_data.size + 1
    assert np.isclose(np.mean(output_buf.real), 0), f"{np.mean(output_buf.real)}"
    assert np.isclose(np.mean(output_buf.imag), 0), f"{np.mean(output_buf.imag)}"
    assert np.isclose(np.std(output_buf.real), target_input_rms)
    assert np.isclose(np.std(output_buf.imag), target_input_rms)


def test_fast_preprocess_single_norm_with_data():
    input_data = np.zeros_like(global_input_data, dtype=np.complex64) + (1+1j)
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    Ai = np.zeros(1, dtype=np.complex64)
    Qi = np.zeros(2, dtype=np.float64)
    N = np.ones(1, dtype=np.int32) 
    calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    fast_preprocess_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms, sky_sub)
    assert np.isclose(Ai.real, 1)
    assert np.isclose(Ai.imag, 1)
    assert np.isclose(Qi[0], 0)
    assert np.isclose(Qi[0], 0)
    assert N == input_data.size + 1
    assert np.isclose(np.mean(output_buf.real), 0)
    assert np.isclose(np.mean(output_buf.imag), 0)
    assert np.isclose(np.std(output_buf.real), target_input_rms)
    assert np.isclose(np.std(output_buf.imag), target_input_rms)






