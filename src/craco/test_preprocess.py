import logging
import numpy as np
from craco.preprocess import fast_preprocess, fast_preprocess_single_norm, fast_preprocess_multi_mean_single_norm, fast_preprocess_sos, fast_cas_crs
from craco.vis_subtractor import VisSubtractor
from craco.timer import Timer
from craft import uvfits, craco_plan
from craco import uvfits_meta, calibration
from pytest import fixture
import os

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
    import pdb
    #pdb.set_trace()
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
    #import pdb
    #pdb.set_trace()
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

# add this so pytest actually detects the tests without breaking. 
# this is dangeours but otherwise we can't run detect tests in VSCODE
# TODO: Use pytest.fixture properly
# ALSO TODO: FInd a way of storing test data properly.
f = None
if os.path.exists(fname): 
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

'''
Ai = np.zeros((nbl, nf), dtype=np.complex64)
Qi = np.zeros((nbl, nf), dtype=np.complex64)
N = np.ones((nbl, nf), dtype=np.int16)

cas = np.zeros((nf, nt), dtype=np.float64)
crs = np.zeros((nf, nt), dtype=np.float64)
cas_N = np.zeros((nf, nt), dtype=np.int16)
'''
cal= calsoln.solarray.mean(axis=2).squeeze()
calsoln_data = cal.data
calsoln_mask = cal.mask


    
def notest_fast_preprocess_single_norm_with_zero():
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


def notest_fast_preprocess_single_norm_with_ones():
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
    assert np.isclose(np.std(output_buf.real), 0)
    assert np.isclose(np.std(output_buf.imag), 0)


def notest_fast_preprocess_single_norm_with_data():
    input_data = global_input_data.copy()
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    Ai = np.zeros(1, dtype=np.complex64)
    Qi = np.zeros(2, dtype=np.float64)
    N = np.ones(1, dtype=np.int32) 
    #calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    expected_mean = np.mean(input_data * calsoln_data[:, :, np.newaxis])
    expected_std = np.std(input_data * calsoln_data[:, :, np.newaxis]) / np.sqrt(2)
    expected_final_mean = 0 + 0j

    fast_preprocess_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms, sky_sub)

    measured_std = np.sqrt((Qi[0] + Qi[1])/ 2 / (N[0]-1 ))
    print(measured_std, N, N[0], Qi, Qi[0], np.sqrt(Qi[0] / (N[0] - 1)))

    assert np.isclose(Ai.real[0], expected_mean.real, rtol=0.001, atol=0.1)
    assert np.isclose(Ai.imag[0], expected_mean.imag, rtol=0.001, atol=0.1)
    assert np.isclose(measured_std, expected_std, rtol=0.001, atol=0.1)
    assert N == input_data.size + 1
    assert np.isclose(np.mean(output_buf.real), expected_final_mean.real, rtol=0.001, atol=0.1)
    assert np.isclose(np.mean(output_buf.imag), expected_final_mean.imag, rtol=0.001, atol=0.1)
    assert np.isclose(np.std(output_buf) / np.sqrt(2), target_input_rms, rtol=0.001, atol=0.1)

def no_test_fast_preprocess_with_single_norm_equality_with_old_function():
    block0.mask = False     #Remove all masks from block0, so that we can compare apples to apples
    nbl = 2
    original_calibrated_output = original_calibrate_input(solarray = cal[2:4, :, np.newaxis], values_subtract = nt, values_target_input_rms = values.target_input_rms, input_flat_raw = block0[2:4])
    input_data = global_input_data.copy()[2:4]
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    global_mean = np.zeros(1, dtype=np.complex64)
    global_Q = np.zeros(2, dtype=np.float64)
    global_N = np.ones(1, dtype=np.int32) 
    
    Ai = np.zeros((nbl, nf), dtype=np.complex64)
    N = np.ones((nbl, nf), dtype=np.int32)

    target_input_rms = values.target_input_rms
    sky_sub = True

    expected_mean = np.mean(input_data)
    expected_std = np.std(input_data) / np.sqrt(2)
    expected_final_mean = 0 + 0j

    fast_preprocess_multi_mean_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, global_mean, global_Q, N, global_N, calsoln_data[2:4], target_input_rms, sky_sub)

    print(output_buf[0, 0, 0], original_calibrated_output.data[0, 0, 0])
    assert np.all(np.isclose(output_buf, original_calibrated_output.data, atol = 0.1, rtol = 0.01))


def test_fast_preprocess_sos_with_zero():
    input_data = np.zeros_like(global_input_data, dtype=np.complex64)
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    s1 = np.zeros((nbl, nf), dtype=np.complex128)
    s2 = np.zeros((2, nbl, nf), dtype=np.float64)
    N = np.ones((nbl, nf), dtype=np.int32)
    calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    fast_preprocess_sos(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, s1, s2, N, calsoln_data, target_input_rms, sky_sub)
    assert np.all(np.isclose(s1.real, 0))
    assert np.all(np.isclose(s1.imag, 0))
    assert np.all(np.isclose(s2[0], 0))
    assert np.all(np.isclose(s2[1], 0))
    assert N.sum() == input_data.size
    assert np.all(output_buf.real == 0)
    assert np.all(output_buf.imag == 0)

def test_fast_preprocess_sos_with_ones():
    input_data = np.zeros_like(global_input_data, dtype=np.complex64) + (1+1j)
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    s1 = np.zeros((nbl, nf), dtype=np.complex128)
    s2 = np.zeros((2, nbl, nf), dtype=np.float64)
    N = np.ones((nbl, nf), dtype=np.int32)
    calsoln_data = np.ones((nbl, nf), dtype=np.complex64)
    target_input_rms = 512
    sky_sub = True

    fast_preprocess_sos(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, s1, s2, N, calsoln_data, target_input_rms, sky_sub)
    assert np.all(np.isclose(s1.real, nt))
    assert np.all(np.isclose(s1.imag, nt))
    assert np.all(np.isclose(s2[0], nt))
    assert np.all(np.isclose(s2[1], nt))
    assert N.sum() == input_data.size
    assert np.isclose(np.mean(output_buf.real), 0), f"{np.mean(output_buf.real)}"
    assert np.isclose(np.mean(output_buf.imag), 0), f"{np.mean(output_buf.imag)}"
    assert np.isclose(np.std(output_buf.real), 0)
    assert np.isclose(np.std(output_buf.imag), 0)


def test_fast_preprocess_sos_with_old_function():
    block0.mask = False     #Remove all masks from block0, so that we can compare apples to apples
    #nbl = 2
    original_calibrated_output = original_calibrate_input(solarray = cal[:, :, np.newaxis], values_subtract = nt, values_target_input_rms = values.target_input_rms, input_flat_raw = block0)
    input_data = global_input_data.copy()
    output_buf = np.zeros_like(input_data)
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    isubblock = 0
    s1 = np.zeros((nbl, nf), dtype=np.complex128)
    s2 = np.zeros((2, nbl, nf), dtype=np.float64)
    N = np.ones((nbl, nf), dtype=np.int32)

    target_input_rms = values.target_input_rms
    sky_sub = True

    expected_mean = np.mean(input_data)
    expected_std = np.std(input_data) / np.sqrt(2)
    expected_final_mean = 0 + 0j
    fast_preprocess_sos(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, s1, s2, N, calsoln_data, target_input_rms, sky_sub)
    
    assert np.all(np.isclose(output_buf, original_calibrated_output.data, atol = 0.01, rtol = 0.001))


def test_fast_cas_crs():
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    cas = np.zeros((nf, nt), dtype=np.float64)
    crs = np.zeros((nf, nt), dtype=np.float64)
    fast_cas_crs(block0.data, bl_weights, fixed_freq_weights, input_tf_weights, cas, crs)

    actual_cas = (block0.data.real**2 + block0.data.imag**2).sum(axis=0)
    actual_crs = (block0.data.real**2).sum(axis=0)
    np.allclose(actual_cas, cas)
    np.allclose(actual_crs, crs)

def test_fast_cas_crs_with_zeros():
    fixed_freq_weights = np.ones(nf, dtype=np.bool)
    bl_weights = np.ones(nbl, dtype=np.bool)
    input_tf_weights = np.ones((nf, nt), dtype=np.bool)
    cas = np.zeros((nf, nt), dtype=np.float64)
    crs = np.zeros((nf, nt), dtype=np.float64)
    fast_cas_crs(block0.data, bl_weights, fixed_freq_weights, input_tf_weights, cas, crs)


    input_data = np.zeros_like(block0.data)
    actual_cas = (input_data.real**2 + input_data.imag**2).sum(axis=0)
    actual_crs = (input_data.real**2).sum(axis=0)
    np.allclose(actual_cas, cas)
    np.allclose(actual_crs, crs)
