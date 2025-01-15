from . import calibration
from iqrm import iqrm_mask
import numpy as np
import warnings, pdb, os
from craft.craco import bl2ant, bl2array
from craft import craco, sigproc
from numba import njit, jit, prange
import IPython

def get_isMasked_nPol(block):

    if type(block) == dict:
        data = block[next(iter(block))]
        ndim_offset = 1
    else:
        data = block
        ndim_offset = 0

    if type(data) == np.ndarray:
        isMasked = False
        ndim = data.ndim
    elif type(data) == np.ma.core.MaskedArray:
        isMasked = True
        ndim = data.ndim
    else:
        raise Exception(f"Expected to get a numpy.ndarray or np.ma.core.MaskedArray, but got {type(block)} instead")

    if ndim + ndim_offset == 3:
        nPol = 0    #nPol 0 means the polarisation axis is missing
    elif ndim + ndim_offset== 4:
        nPol = data.shape[-2]
    else:
        raise Exception(f"ndim of a single baseline can only be 2 or 3, found {ndim}")
    return isMasked, nPol, type(block)





def normalise(block, target_input_rms = 1):
    '''
    Normalises each baseline and channel in the visibility block to
    0 median and a desired std value along the time axis.
    It modifies the array/dictionary in place

    Input
    -----
    block: np.ndarray or np.ma.masked_array or dict
            Input visbility data of shape (nbl, nf, [npol], nt) if array, or,
            Visibility dict with nbl arrays/masked arrays of 
            (nf, [npol], nt) shape each. The npol axis is optional

    target_input_rms: float
            Desired std of each channel
    '''
    
    if type(block) == dict:
        new_block = {}
        for ibl, bldata in block.items():
            #print(f"====>> The shape of received block[ibl] for ibl{ibl} is {block[ibl].shape}")
            existing_rms = bldata.std(axis=-1) / np.sqrt(2)
            new_block[ibl] = (bldata - np.mean(bldata, axis=-1, keepdims=True)) * (target_input_rms / existing_rms)[..., None]
            new_block[ibl] = new_block[ibl].astype(bldata.dtype)
            #print(f"====>> The shape of normalised block[ibl] for ibl{ibl} is {block[ibl].shape}")
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        existing_rms = block.std(axis=-1) / np.sqrt(2)   
        new_block = (block - np.mean(block, axis=-1, keepdims = True)) * (target_input_rms / existing_rms)[..., None]
        new_block = new_block.astype(block.dtype)

    else:
        raise Exception("Unknown type of block provided - expecting dict, np.ndarray, or np.ma.core.MaskedArray")

    return new_block


def average_pols(block, keepdims = True):
    '''
    Average the pol axis and remove the extra dimension. Edits it in-place

    Input
    -----
    block: dict or np.ndarray or np.ma.core.MaskedArray
            The block containing the data. 
            If dict, it should be keyed by blid and each value
            should contain an array of shape (nf, npol, nt)
            If array (regular or masked) it should have the shape
            (nbl, nf, npol, nt)

    '''
    _, nPol, _ = get_isMasked_nPol(block)
    if nPol > 1:
        raise RuntimeError(f"AAaaaaaaah! I cannot process blocks with nPol > 1, found - {nPol}")
    if keepdims == True:
        #This is under the assumption that there is only 1 pol in the block
        return block

    if nPol == 0:
        print("No pol axis to scrunch, returning the block as it is")
        return block
    if type(block) == dict:
        for ibl, bldata in block.items():
            assert bldata.ndim == 3, f"Exptected 3 dimensions (nf, npol, nt), but got {bldata.ndim}, shape={bldata.shape}"#, block = {block},\n bldata= {bldata}"
            block[ibl] = bldata[:, 0, ...]
            #Commenting out the lines belwo assuming there will only be 1 pol in the data, see CRACO-154
            #block[ibl] = bldata.mean(axis=1, keepdims=keepdims)
            #if not keepdims:
            #    block[ibl] = block[ibl].squeeze()
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        assert block.ndim == 4, f"Expected 4 dimensions (nbl, nf, npol, nt), but got {bldata.ndim}"
        block = block[:, :, 0, :]
        #Commenting out the lines below assuming that there will only be 1 pol in the data, see CRACO-154
        #block = block.mean(axis=2, keepdims=keepdims)
        #if not keepdims:
        #    block = block.squeeze()
    return block


 
@njit(parallel=False, cache=True)
def fast_cas_crs(input_data, bl_weights, fixed_freq_weights, input_tf_weights, cas, crs):
    '''
    input_block: Input data array - (nbl, nf, nt). np.ndarray - complex64
    bl_weights: Input mask array - (nbl - boolean; 0 for flagged, 1 for good
    fixed_freq_weights: numpy array containing the fixed freq weights (nf) - boolean; 0 for flagged, 1 for good
    input_tf_weights: numpy array containing 2D tf weights (nf, nt) - boolean; 0 for flagged, 1 for good
    cas: Sum of amplitude of all baselines - unmasked numpy array (nf, nt) - float
    crs: Sum of real part of all baselines - unmasked numpy array (nf, nt) - float
    '''
    nbl, nf, nt = input_data.shape
    for i_bl in range(nbl):
        if bl_weights[i_bl] == 0:
            continue        #If all baselines happen to be flagged, then we don't touch cas and crs, which means the values may be garbage!!! TODO - fix this by ensuring cas and crs are 0s
        for i_f in range(nf):
            if fixed_freq_weights[i_f] == 0:
                cas[i_f, :] = 0
                crs[i_f, :] = 0
            #If Keith can gaurantee that input_data will be zeroed whenever there is a packet drop or metadata is flagged, then I don't need to multiply by the input_weights here
            channel_baseline = input_data[i_bl, i_f, :] * input_tf_weights[i_f, :]

            if i_bl == 0:
                cas[i_f, :] = channel_baseline.real**2 + channel_baseline.imag**2
                crs[i_f, :] = channel_baseline.real**2
            else:
                cas[i_f, :] += channel_baseline.real**2 + channel_baseline.imag**2
                crs[i_f, :] += channel_baseline.real**2


def get_simple_dynamic_rfi_masks(cas, crs, finest_nt, tf_weights, freq_radius, freq_threshold, use_crs = False):
    '''
    cas - (nf, nt) ndarray float 64
    crs - (nf, nt) ndarray float 64
    finest_nt - finest timescale (in samples) on which rms needs to be measured to do RFI mitigation
    tf_weights - (nf, nt) - ndarray boolean, the output mask array derived from cas and crs only.
                - Either an empty boolean (initialed to 1s) can be passed to this function, or the input_tf_masks (coming from dropped packets) can be passed, which will then automatically merged with the dynamic RFI masks here
    freq_radius - int, units of nchan
    freq_threshold - float, threshold above which the cas_rmses would get zapped
    '''
    nf, nt = cas.shape
    assert nt % finest_nt == 0, "nt has to be a integral multiple of finest_nt"

    nsubblock = int(nt / finest_nt)

    for isubblock in range(nsubblock):
        cas_rmses = cas[:, isubblock*finest_nt:(isubblock+1)*finest_nt].std(axis=-1)
        if use_crs:
            crs_rmses = crs[:, isubblock*finest_nt:(isubblock+1)*finest_nt].std(axis=-1)

        freq_flag_cas, _ = iqrm_mask(cas_rmses, freq_radius, freq_threshold)
        if use_crs:
            freq_flag_crs, _ = iqrm_mask(crs_rmses, freq_radius, freq_threshold)

        tf_weights[freq_flag_cas, isubblock*finest_nt:(isubblock+1)*finest_nt] = 0
        if use_crs:
            tf_weights[freq_flag_crs, isubblock*finest_nt:(isubblock+1)*finest_nt] = 0



def get_complicated_dynamic_rfi_masks(cas, crs, finest_nt, rmses, Ai, Qi, N):
    nf, nt = cas.shape
    assert nt % finest_nt == 0 and (nt / finest_nt) % 2 == 0, "nt has to be a multiple in power of 2 of finest_nt, i.e. nt = finest_nt * 2^x"

    nsubblocks = nt / finest_nt

    #Plan to do multi-layer rms computation where we measure the rms over different time scales
        

@njit(cache=True)
def get_subblock_rms(input_data, rmses, bl_weights, tf_weights, isubblock, finest_nt, nbl, nf, nt):
    '''
    Computes the rms (squared) along the time axis (across all baselines) as a function of frequency
    Saves the output in the rmses array. Does the computation on a subblock of data defined by finest_nt and isubblock
    '''
    for i_f in prange(nf):
        for i_bl in range(nbl):
            
            if bl_weights[i_bl] == 0:
                continue
                
            for i_t in range(finest_nt):
                if tf_weights[i_f, isubblock * finest_nt + i_t] == 0:
                    continue

                v = input_data[i_bl, i_f, isubblock * finest_nt + i_t]
                rmses[i_f] += v.real*v.real + v.imag*v.imag

def get_phase_varying_dynamic_rfi_masks(input_data, finest_nt, bl_weights, tf_weights, freq_radius, freq_threshold):
    '''
    input_data - np.ndarray - complex64
                Numpy array containing the visibility data
    finest_nt - int
                finest timescale (in samples) on which rms needs to be measured to do RFI mitigation
    bl_weights - np.ndarray - boolean
                Numpy array of shape nbl containing the baseline weights
    tf_weights - (nf, nt) - np.ndarray - boolean
                Numpy array containing the input_tf_weights
                This array will get modified in-place and newly identified bad channels will be set to 0 weight
    freq_radius - int
                Radius of influence to consider when finding outliers in freq (units of nchan)
    freq_threshold - float
                Threshold above which the channels with higher rmses would get zapped (units of sigma)
    '''
    nbl, nf, nt = input_data.shape
    assert nt % finest_nt == 0, "nt has to be a integral multiple of finest_nt"
    rmses = np.zeros(nf, dtype='float64')
    nsubblock = int(nt / finest_nt)
    for isubblock in prange(nsubblock):
        rmses[:] = 0
        get_subblock_rms(input_data, rmses, bl_weights, tf_weights, isubblock, finest_nt, nbl, nf, nt)
        
        freq_flags, _ = iqrm_mask(rmses, freq_radius, freq_threshold)
        tf_weights[freq_flags, isubblock*finest_nt:(isubblock+1)*finest_nt] = 0
        
        


#import pdb
@njit(parallel=True, cache=True)
def fast_preprocess(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms=None, sky_sub = False):
    '''
    Loops over all dimensions of the input_block. Applies the calibration soln,
    Measures the input levels, calculates cas/crs and optionally rescales and does the sky subtraction.

    Ideally, one should just measure the cas and crs in the first pass.
    Then get new masks and use them to mask bad data in the 2nd pass, while accumulating the rms statistics.
    And then apply the measured rms values in the normalisation pass (3rd).
    
    But if you assume that the data would be flagged for the entire block at a time, then one can calculate the RFI masks after normalisation 
    This is because data for which improper rms values would have been computed due to RFI will eventually get masked, so we don't care.
    This allows us to process the data in just 2 passes.

    you would want to just measure the statistics and cas/crs through this function.
    Then you should get the masks using the cas and crs. Then re-apply the 

    
    input_block: Input data array - (nbl, nf, nt). np.ndarray - complex64
    bl_weights: Input weights for baselines - (nbl) ndarray - boolean; 0 means bad, 1 means good
    input_tf_weights: Input weights for tf - (nf, nt) [For example due to dropped packets] ndarray - boolean; 0 means bad, 1 means good
    fixed_freq_weights: Fixed - (nf). np.ndarray - boolean; 0 for bad, 1 for good
    output_buf: Output data array - (nbl, nf, nt*nsubblock). Won't be a masked array - complex64
    isubblock: integer - output_buf can hold mulitple input_blocks - isubblock is the counter
    Ai: numpy array to keep track of the running mean (nbl, nf) - complex64
    Qi: numpy array to keep track of the running rms (nbl, nf) - complex64
    N: numpy array to keep track of the nnumbers added so far (nbl, nf) - int16
        This needs to be initialised with ones at the very beginning, otherwise bad things will happen
    calsoln_data: numpy array containing the calibration solution (nbl, nf) - complex64
    calsoln_mask: numpy array containing the calibration masks (nbl, nf) - boolean
    cas: Sum of amplitude of all baselines - unmasked numpy array (nf, nt) - float
    crs: Sum of real part of all baselines - unmasked numpy array (nf, nt) - float
    cas_N: numpy array to keep track of the numbers added into cas and crs arrays so far (nf, nt) - int16 (must be initialed to 0s)
    target_input_rms: float, the rms value which needs to be set. If None, data will not be rescaled
    sky_sub: bool, Whether to do mean subtraction or not.
    reset_scales: bool, whether to keep on accumulating Ai and Qi values across blocks, or reset to zero at the start of this block
                    Note -- reset_scales flag should set to true when this function is called for the first time, or we need to ensure that
                    Ai, Qi and N are initialised properly outside this function.
    '''
    #TODO -- think about whether changing the order of the input_block could speed things up a bit more
    #TODO -- include the computation of CAS and the masks inside this function's first pass over the data
    #        Right now it is a bit too complicated to allow for a different nt for the flagger
    nbl, nf, nt = input_data.shape
    '''
    if input_mask is None:
        #assert type(input_block)==np.ma.core.MaskedArray, f"Given - {type(input_block)}"
        input_data = input_block.data
        #input_mask = np.any(input_block.mask, axis=0)   #We take the OR of all mask values along the Basline axis given that Keith says that if a packet is dropped, all baselines are gauranteed to be flagged for that time-freq spaxel
        input_mask = input_block.mask

    else:
        input_mask = input_mask
        #assert input_mask.shape == input_block.shape, f"Shape of the input masks should match that of input_block - Input_block.shape = {input_block.shape}, Given shape = {input_mask.shape}"

        if type(input_block) == np.ma.core.MaskedArray:
            input_data = input_block.data
        elif type(input_block) == np.ndarray:
            input_data = input_block
        else:
            #raise ValueError(f"dtype of input block should be np.ma.core.MaskedArray, or nd.ndarray, found = {type(input_block)}")
            return 1
    '''
    input_data = input_data
    reset_scales = isubblock==0
    #assert type(calsoln)==np.ma.core.MaskedArray, f"Given - {type(calsoln)}"
 
    cal_data = calsoln_data

    if target_input_rms is None:
        target_rms_value = 0
        apply_rms = False
    else:
        target_rms_value = target_input_rms     #I am copying the value in a new variable because numba doesn't like if a variable which can be None, has a multiplication statement within a conditional block.
        apply_rms = True

    for i_bl in range(nbl):

        if bl_weights[i_bl] == 0:
            output_buf[i_bl, :, isubblock*nt:(isubblock+1)*nt] = 0

            if reset_scales:
                Ai[i_bl] = 0
                Qi[i_bl] = 0
                N[i_bl] = 1
            else:
                #If we are not resetting scales in this block, then we basically leave all Ai, Qis and Ns untouched
                pass
            
            continue
                
        for i_f in range(nf):
            if reset_scales:
                Qi[i_bl, i_f] = 0
                Ai[i_bl, i_f] = 0
                N[i_bl, i_f] = 1

            if fixed_freq_weights[i_f] == 0:
                output_buf[i_bl, i_f, isubblock*nt:(isubblock+1)*nt] = 0
                continue
            
            for i_t in range(nt):
                isamp = input_data[i_bl, i_f, i_t]
                imask = input_tf_weights[i_f, i_t] == 0

                if imask:
                    output_buf[i_bl, i_f, isubblock*nt + i_t] = 0

                else:
                    Qi.real[i_bl, i_f] += (N[i_bl, i_f] -1)/ N[i_bl, i_f] * (isamp.real - Ai[i_bl, i_f].real)**2
                    Qi.imag[i_bl, i_f] += (N[i_bl, i_f] -1)/ N[i_bl, i_f] * (isamp.imag - Ai[i_bl, i_f].imag)**2
                    Ai[i_bl, i_f] += (isamp - Ai[i_bl, i_f])/ N[i_bl, i_f]
                    N[i_bl, i_f] += 1

                    #pdb.set_trace()
                    if not apply_rms and not sky_sub:
                        #We don't need to apply any kind of normalisation, so we can apply the cal right inside the first loop
                        #print(f"shapes of output_buf = {output_buf.shape}, cal_data = {cal_data.shape}")
                        output_buf[i_bl, i_f, isubblock*nt + i_t] = isamp * cal_data[i_bl, i_f]

            if apply_rms or sky_sub:
                #pdb.set_trace()
                '''
                rms_val_real = np.sqrt(Qi[i_bl, i_f].real / N[i_bl, i_f])
                rms_val_imag = np.sqrt(Qi[i_bl, i_f].imag / N[i_bl, i_f])
                rms_val = np.sqrt(rms_val_real**2 + rms_val_imag**2) / np.sqrt(2)
                '''
                if N[i_bl, i_f] == 1:
                    continue    #This can happen when the whole channel is flagged, so we'll get a division by zero error in the line below

                rms_val = np.sqrt( (Qi[i_bl, i_f].real + Qi[i_bl, i_f].imag) / (2 * (N[i_bl, i_f]-1) ) ).astype(np.float32) 

                #pdb.set_trace()
                if rms_val == 0:
                    #rms_val can be zero if the channel was zapped by the dynamic RFI flagger
                    #We can't assume that the flagging with zap the channel for all times in this block
                    #So we have to save those masks in a time dependent way (i.e. in the input_tf_weights)
                    #And therefore we can't just skip over those channels without entering the time loop and reaching this stage of applying_rms or subtracting the sky
                    #So we have to deal with such cases here, by simply saying continue, as the output data has already been zeroed.
                    #Otherwise, we will get annoying divison by zero errors when we compute the multiplier
                    continue
                
                calval = cal_data[i_bl, i_f]
                multiplier = calval * target_rms_value  / (rms_val * np.abs(calval))
                #pdb.set_trace()
                #Loop through the data again and apply the various rescale factors and the cal soln
                for i_t in range(nt):
                    isamp = input_data[i_bl, i_f, i_t]
                    imask = input_tf_weights[i_f, i_t] == 0

                    if imask:
                        #Output has already been set to zero in the first loop. 
                        #We don't need to do anything here
                        pass
                    else:
                        if sky_sub and not apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp -  Ai[i_bl, i_f]) * calval

                        elif sky_sub and apply_rms:
                            #pdb.set_trace()
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp - Ai[i_bl, i_f]) * multiplier

                        elif apply_rms and not sky_sub:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp - Ai[i_bl, i_f]) * multiplier + Ai[i_bl, i_f] * calval
                
#@njit(parallel=True, cache=True)
def fast_preprocess_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, Qi, N, calsoln_data, target_input_rms=None, sky_sub = False):
    '''
    Loops over all dimensions of the input_block. Applies the calibration soln,
    Measures the input levels, calculates cas/crs and optionally rescales and does the sky subtraction.

    Ideally, one should just measure the cas and crs in the first pass.
    Then get new masks and use them to mask bad data in the 2nd pass, while accumulating the rms statistics.
    And then apply the measured rms values in the normalisation pass (3rd).
    
    But if you assume that the data would be flagged for the entire block at a time, then one can calculate the RFI masks after normalisation 
    This is because data for which improper rms values would have been computed due to RFI will eventually get masked, so we don't care.
    This allows us to process the data in just 2 passes.

    you would want to just measure the statistics and cas/crs through this function.
    Then you should get the masks using the cas and crs. Then re-apply the 

    
    input_block: Input data array - (nbl, nf, nt). np.ndarray - complex64
    bl_weights: Input weights for baselines - (nbl) ndarray - boolean; 0 means bad, 1 means good
    input_tf_weights: Input weights for tf - (nf, nt) [For example due to dropped packets] ndarray - boolean; 0 means bad, 1 means good
    fixed_freq_weights: Fixed - (nf). np.ndarray - boolean; 0 for bad, 1 for good
    output_buf: Output data array - (nbl, nf, nt*nsubblock). Won't be a masked array - complex64
    isubblock: integer - output_buf can hold mulitple input_blocks - isubblock is the counter
    Ai: 1 element array - complex64
    Qi: 1 element array - complex64
    N: const - int16
        This needs to be initialised with ones at the very beginning, otherwise bad things will happen
    calsoln_data: numpy array containing the calibration solution (nbl, nf) - complex64
    calsoln_mask: numpy array containing the calibration masks (nbl, nf) - boolean
    cas: Sum of amplitude of all baselines - unmasked numpy array (nf, nt) - float
    crs: Sum of real part of all baselines - unmasked numpy array (nf, nt) - float
    cas_N: numpy array to keep track of the numbers added into cas and crs arrays so far (nf, nt) - int16 (must be initialed to 0s)
    target_input_rms: float, the rms value which needs to be set. If None, data will not be rescaled
    sky_sub: bool, Whether to do mean subtraction or not.
    reset_scales: bool, whether to keep on accumulating Ai and Qi values across blocks, or reset to zero at the start of this block
                    Note -- reset_scales flag should set to true when this function is called for the first time, or we need to ensure that
                    Ai, Qi and N are initialised properly outside this function.
    '''
    #TODO -- think about whether changing the order of the input_block could speed things up a bit more
    #TODO -- include the computation of CAS and the masks inside this function's first pass over the data
    #        Right now it is a bit too complicated to allow for a different nt for the flagger
    nbl, nf, nt = input_data.shape
    '''
    if input_mask is None:
        #assert type(input_block)==np.ma.core.MaskedArray, f"Given - {type(input_block)}"
        input_data = input_block.data
        #input_mask = np.any(input_block.mask, axis=0)   #We take the OR of all mask values along the Basline axis given that Keith says that if a packet is dropped, all baselines are gauranteed to be flagged for that time-freq spaxel
        input_mask = input_block.mask

    else:
        input_mask = input_mask
        #assert input_mask.shape == input_block.shape, f"Shape of the input masks should match that of input_block - Input_block.shape = {input_block.shape}, Given shape = {input_mask.shape}"

        if type(input_block) == np.ma.core.MaskedArray:
            input_data = input_block.data
        elif type(input_block) == np.ndarray:
            input_data = input_block
        else:
            #raise ValueError(f"dtype of input block should be np.ma.core.MaskedArray, or nd.ndarray, found = {type(input_block)}")
            return 1
    '''
    input_data = input_data
    if isubblock==0:
        Ai[0] = 0j
        Qi[0] = 0
        Qi[1] = 0
        N[0] = 1
    #assert type(calsoln)==np.ma.core.MaskedArray, f"Given - {type(calsoln)}"
 
    cal_data = calsoln_data

    if target_input_rms is None:
        target_rms_value = 0
        apply_rms = False
    else:
        target_rms_value = target_input_rms     #I am copying the value in a new variable because numba doesn't like if a variable which can be None, has a multiplication statement within a conditional block.
        apply_rms = True

    for i_bl in range(nbl):

        if bl_weights[i_bl] == 0:
            output_buf[i_bl, :, isubblock*nt:(isubblock+1)*nt] = 0
            continue
                
        for i_f in range(nf):
            if fixed_freq_weights[i_f] == 0:
                output_buf[i_bl, i_f, isubblock*nt:(isubblock+1)*nt] = 0
                continue
            cal_samp = cal_data[i_bl, i_f]
            for i_t in range(nt):
                isamp = input_data[i_bl, i_f, i_t]
                imask = input_tf_weights[i_f, i_t] == 0

                if imask:
                    output_buf[i_bl, i_f, isubblock*nt + i_t] = 0

                else:
                    #There is a bug in numba where I cannot set complex_var(128bit).real = real_var(64bit)
                    #See https://github.com/numba/numba/issues/3573

                    resulting_samp = isamp * cal_samp
                    Qi[0] += (N[0] -1)/ N[0] * (resulting_samp.real - Ai[0].real)**2
                    Qi[1] += (N[0] -1)/ N[0] * (resulting_samp.imag - Ai[0].imag)**2
                    Ai[0] += (resulting_samp - Ai[0])/ N[0]
                    N[0] += 1

                    #pdb.set_trace()
                    if not apply_rms and not sky_sub:
                        #We don't need to apply any kind of normalisation, so we can apply the cal right inside the first loop
                        #print(f"shapes of output_buf = {output_buf.shape}, cal_data = {cal_data.shape}")
                        output_buf[i_bl, i_f, isubblock*nt + i_t] = resulting_samp

    if apply_rms or sky_sub:

        #pdb.set_trace()
        '''
        rms_val_real = np.sqrt(Qi[i_bl, i_f].real / N[i_bl, i_f])
        rms_val_imag = np.sqrt(Qi[i_bl, i_f].imag / N[i_bl, i_f])
        rms_val = np.sqrt(rms_val_real**2 + rms_val_imag**2) / np.sqrt(2)
        '''

        if N[0] == 1:
            return  #All data must have been flagged. If I don't return here, then we get a division by zero error in the line below
        
        rms_val = np.sqrt( (Qi[0] + Qi[1]) / (2 * (N[0]-1)) )

        if rms_val == 0:
            #rms_val can be zero if the channel was zapped by the dynamic RFI flagger
            #We can't assume that the flagging with zap the channel for all times in this block
            #So we have to save those masks in a time dependent way (i.e. in the input_tf_weights)
            #And therefore we can't just skip over those channels without entering the time loop and reaching this stage of applying_rms or subtracting the sky
            #So we have to deal with such cases here, by simply saying continue, as the output data has already been zeroed.
            #Otherwise, we will get annoying divison by zero errors when we compute the multiplier
            
            return #continue
        
        for i_bl in range(nbl):
            if bl_weights[i_bl] == 0:
                continue
            for i_f in range(nf):    
                if fixed_freq_weights[i_f] == 0:
                    continue            

                calval = cal_data[i_bl, i_f]
                #multiplier = calval * target_rms_value  / (rms_val)# * np.abs(calval))
                multiplier = target_rms_value  / (rms_val)# * np.abs(calval))
                pdb.set_trace()
                #Loop through the data again and apply the various rescale factors and the cal soln
                for i_t in range(nt):
                    isamp = input_data[i_bl, i_f, i_t]
                    imask = input_tf_weights[i_f, i_t] == 0

                    if imask:
                        #Output has already been set to zero in the first loop. 
                        #We don't need to do anything here
                        pass
                    else:
                        if sky_sub and not apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval -  Ai[0])

                        elif sky_sub and apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval - Ai[0]) * multiplier

                        elif apply_rms and not sky_sub:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval - Ai[0]) * multiplier + Ai[0] / calval
                

#@njit(parallel=True, cache=True)
def fast_preprocess_multi_mean_single_norm(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, Ai, global_mean, global_Q, N, global_N, calsoln_data, target_input_rms=None, sky_sub = False):
    '''
    Loops over all dimensions of the input_block. Applies the calibration soln,
    Measures the input levels, calculates cas/crs and optionally rescales and does the sky subtraction.

    Ideally, one should just measure the cas and crs in the first pass.
    Then get new masks and use them to mask bad data in the 2nd pass, while accumulating the rms statistics.
    And then apply the measured rms values in the normalisation pass (3rd).
    
    But if you assume that the data would be flagged for the entire block at a time, then one can calculate the RFI masks after normalisation 
    This is because data for which improper rms values would have been computed due to RFI will eventually get masked, so we don't care.
    This allows us to process the data in just 2 passes.

    you would want to just measure the statistics and cas/crs through this function.
    Then you should get the masks using the cas and crs. Then re-apply the 

    
    input_block: Input data array - (nbl, nf, nt). np.ndarray - complex64
    bl_weights: Input weights for baselines - (nbl) ndarray - boolean; 0 means bad, 1 means good
    input_tf_weights: Input weights for tf - (nf, nt) [For example due to dropped packets] ndarray - boolean; 0 means bad, 1 means good
    fixed_freq_weights: Fixed - (nf). np.ndarray - boolean; 0 for bad, 1 for good
    output_buf: Output data array - (nbl, nf, nt*nsubblock). Won't be a masked array - complex64
    isubblock: integer - output_buf can hold mulitple input_blocks - isubblock is the counter
    Ai: (nbl, nf) shaped array - complex64
    global_mean: 1 element array - complex64
    global_Q: 2 element array - float32
    N: (nbl, nf) - int32
    global_N: const - int32
        These need to be initialised with ones at the very beginning, otherwise bad things will happen
    calsoln_data: numpy array containing the calibration solution (nbl, nf) - complex64
    calsoln_mask: numpy array containing the calibration masks (nbl, nf) - boolean
    cas: Sum of amplitude of all baselines - unmasked numpy array (nf, nt) - float
    crs: Sum of real part of all baselines - unmasked numpy array (nf, nt) - float
    cas_N: numpy array to keep track of the numbers added into cas and crs arrays so far (nf, nt) - int16 (must be initialed to 0s)
    target_input_rms: float, the rms value which needs to be set. If None, data will not be rescaled
    sky_sub: bool, Whether to do mean subtraction or not.
    reset_scales: bool, whether to keep on accumulating Ai and Qi values across blocks, or reset to zero at the start of this block
                    Note -- reset_scales flag should set to true when this function is called for the first time, or we need to ensure that
                    Ai, Qi and N are initialised properly outside this function.
    '''
    #TODO -- think about whether changing the order of the input_block could speed things up a bit more
    #TODO -- include the computation of CAS and the masks inside this function's first pass over the data
    #        Right now it is a bit too complicated to allow for a different nt for the flagger
    nbl, nf, nt = input_data.shape
    '''
    if input_mask is None:
        #assert type(input_block)==np.ma.core.MaskedArray, f"Given - {type(input_block)}"
        input_data = input_block.data
        #input_mask = np.any(input_block.mask, axis=0)   #We take the OR of all mask values along the Basline axis given that Keith says that if a packet is dropped, all baselines are gauranteed to be flagged for that time-freq spaxel
        input_mask = input_block.mask

    else:
        input_mask = input_mask
        #assert input_mask.shape == input_block.shape, f"Shape of the input masks should match that of input_block - Input_block.shape = {input_block.shape}, Given shape = {input_mask.shape}"

        if type(input_block) == np.ma.core.MaskedArray:
            input_data = input_block.data
        elif type(input_block) == np.ndarray:
            input_data = input_block
        else:
            #raise ValueError(f"dtype of input block should be np.ma.core.MaskedArray, or nd.ndarray, found = {type(input_block)}")
            return 1
    '''
    input_data = input_data
    reset_scales = isubblock == 0
    if reset_scales:
        global_mean[0] = 0j
        global_Q[0] = 0
        global_Q[1] = 0
        global_N[0] = 1
 
    cal_data = calsoln_data

    if target_input_rms is None:
        target_rms_value = 0
        apply_rms = False
    else:
        target_rms_value = target_input_rms     #I am copying the value in a new variable because numba doesn't like if a variable which can be None, has a multiplication statement within a conditional block.
        apply_rms = True

    for i_bl in range(nbl):

        if bl_weights[i_bl] == 0:
            if reset_scales:
                Ai[i_bl] = 0j
                N[i_bl] = 1

            output_buf[i_bl, :, isubblock*nt:(isubblock+1)*nt] = 0
            continue
                
        for i_f in range(nf):

            if reset_scales:
                Ai[i_bl, i_f] = 0j
                N[i_bl, i_f] = 1

            if fixed_freq_weights[i_f] == 0:
                output_buf[i_bl, i_f, isubblock*nt:(isubblock+1)*nt] = 0
                continue

            cal_samp = cal_data[i_bl, i_f]
            for i_t in range(nt):
                isamp = input_data[i_bl, i_f, i_t]
                imask = input_tf_weights[i_f, i_t] == 0

                if imask:
                    output_buf[i_bl, i_f, isubblock*nt + i_t] = 0

                else:
                    #There is a bug in numba where I cannot set complex_var(128bit).real = real_var(64bit)
                    #See https://github.com/numba/numba/issues/3573

                    resulting_samp = isamp * cal_samp
                    #test = global_N[0]
                    global_Q[0] += (global_N[0] -1)/ global_N[0] * (resulting_samp.real - Ai[i_bl, i_f].real)**2
                    global_Q[1] += (global_N[0] -1)/ global_N[0] * (resulting_samp.imag - Ai[i_bl, i_f].imag)**2
                    global_mean[0] += (resulting_samp - global_mean[0])/ global_N[0]
                    global_N[0] += 1

                    Ai[i_bl, i_f] += (resulting_samp - Ai[i_bl, i_f])/ N[i_bl, i_f]
                    N[i_bl, i_f] += 1

                    if not apply_rms and not sky_sub:
                        #We don't need to apply any kind of normalisation, so we can apply the cal right inside the first loop
                        #print(f"shapes of output_buf = {output_buf.shape}, cal_data = {cal_data.shape}")
                        output_buf[i_bl, i_f, isubblock*nt + i_t] = resulting_samp
            #pdb.set_trace()

    if apply_rms or sky_sub:

        pdb.set_trace()
        '''
        rms_val_real = np.sqrt(Qi[i_bl, i_f].real / N[i_bl, i_f])
        rms_val_imag = np.sqrt(Qi[i_bl, i_f].imag / N[i_bl, i_f])
        rms_val = np.sqrt(rms_val_real**2 + rms_val_imag**2) / np.sqrt(2)
        '''

        if global_N[0] == 1:
            return  #All data must have been flagged. If I don't return here, then we get a division by zero error in the line below
        
        rms_val = np.sqrt( (global_Q[0] + global_Q[1]) / (2 * (global_N[0]-1)) )

        if rms_val == 0:
            return
        
        for i_bl in range(nbl):
            if bl_weights[i_bl] == 0:
                continue
            for i_f in range(nf):    
                if fixed_freq_weights[i_f] == 0:
                    continue            

                calval = cal_data[i_bl, i_f]
                #multiplier = calval * target_rms_value  / (rms_val)# * np.abs(calval))
                multiplier = target_rms_value  / (rms_val)# * np.abs(calval))
                #pdb.set_trace()
                #Loop through the data again and apply the various rescale factors and the cal soln
                for i_t in range(nt):
                    isamp = input_data[i_bl, i_f, i_t]
                    imask = input_tf_weights[i_f, i_t] == 0

                    if imask:
                        #Output has already been set to zero in the first loop. 
                        #We don't need to do anything here
                        pass
                    else:
                        if sky_sub and not apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval -  Ai[i_bl, i_f])

                        elif sky_sub and apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval - Ai[i_bl, i_f]) * multiplier

                        elif apply_rms and not sky_sub:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp * calval - Ai[i_bl, i_f]) * multiplier + Ai[i_bl, i_f] / calval
                


@njit(parallel=False, cache=True)
def fast_preprocess_sos(input_data, bl_weights, fixed_freq_weights, input_tf_weights, output_buf, isubblock, means, s1, s2, N, calsoln_data, target_input_rms=None, sky_sub = False, global_norm = True):
    '''
    Loops over all dimensions of the input_block. Applies the calibration soln,
    Measures the input levels, calculates cas/crs and optionally rescales and does the sky subtraction.

    Ideally, one should just measure the cas and crs in the first pass.
    Then get new masks and use them to mask bad data in the 2nd pass, while accumulating the rms statistics.
    And then apply the measured rms values in the normalisation pass (3rd).
    
    But if you assume that the data would be flagged for the entire block at a time, then one can calculate the RFI masks after normalisation 
    This is because data for which improper rms values would have been computed due to RFI will eventually get masked, so we don't care.
    This allows us to process the data in just 2 passes.

    you would want to just measure the statistics and cas/crs through this function.
    Then you should get the masks using the cas and crs. Then re-apply the 

    
    input_block: Input data array - (nbl, nf, nt). np.ndarray - complex64
    bl_weights: Input weights for baselines - (nbl) ndarray - boolean; 0 means bad, 1 means good
    input_tf_weights: Input weights for tf - (nf, nt) [For example due to dropped packets] ndarray - boolean; 0 means bad, 1 means good
    fixed_freq_weights: Fixed - (nf). np.ndarray - boolean; 0 for bad, 1 for good
    output_buf: Output data array - (nbl, nf, nt*nsubblock). Won't be a masked array - complex64
    isubblock: integer - output_buf can hold mulitple input_blocks - isubblock is the counter
    s1: (nbl, nf) shaped array - complex64
    s2: (2, nbl, nf) shaped array - float32
    N: (nbl, nf) - int32
    calsoln_data: numpy array containing the calibration solution (nbl, nf) - complex64
    calsoln_mask: numpy array containing the calibration masks (nbl, nf) - boolean
    cas: Sum of amplitude of all baselines - unmasked numpy array (nf, nt) - float
    crs: Sum of real part of all baselines - unmasked numpy array (nf, nt) - float
    cas_N: numpy array to keep track of the numbers added into cas and crs arrays so far (nf, nt) - int16 (must be initialed to 0s)
    target_input_rms: float, the rms value which needs to be set. If None, data will not be rescaled
    sky_sub: bool, Whether to do mean subtraction or not.
    reset_scales: bool, whether to keep on accumulating Ai and Qi values across blocks, or reset to zero at the start of this block
                    Note -- reset_scales flag should set to true when this function is called for the first time, or we need to ensure that
                    Ai, Qi and N are initialised properly outside this function.
    global_norm: bool, whether to compute the rms normalisation factor per baseline-channel, or per block globally. True means globally.
    '''
    #TODO -- think about whether changing the order of the input_block could speed things up a bit more
    #TODO -- include the computation of CAS and the masks inside this function's first pass over the data
    #        Right now it is a bit too complicated to allow for a different nt for the flagger
    nbl, nf, nt = input_data.shape
    '''
    if input_mask is None:
        #assert type(input_block)==np.ma.core.MaskedArray, f"Given - {type(input_block)}"
        input_data = input_block.data
        #input_mask = np.any(input_block.mask, axis=0)   #We take the OR of all mask values along the Basline axis given that Keith says that if a packet is dropped, all baselines are gauranteed to be flagged for that time-freq spaxel
        input_mask = input_block.mask

    else:
        input_mask = input_mask
        #assert input_mask.shape == input_block.shape, f"Shape of the input masks should match that of input_block - Input_block.shape = {input_block.shape}, Given shape = {input_mask.shape}"

        if type(input_block) == np.ma.core.MaskedArray:
            input_data = input_block.data
        elif type(input_block) == np.ndarray:
            input_data = input_block
        else:
            #raise ValueError(f"dtype of input block should be np.ma.core.MaskedArray, or nd.ndarray, found = {type(input_block)}")
            return 1
    '''
    input_data = input_data
    reset_scales = isubblock == 0
    cal_data = calsoln_data

    if target_input_rms is None:
        target_rms_value = 0
        apply_rms = False
    else:
        target_rms_value = target_input_rms     #I am copying the value in a new variable because numba doesn't like if a variable which can be None, has a multiplication statement within a conditional block.
        apply_rms = True

    for i_bl in prange(nbl):

        if bl_weights[i_bl] == 0:
            if reset_scales:
                s1[i_bl] = 0j
                s2[0, i_bl] = 0
                s2[1, i_bl] = 0
                N[i_bl] = 0

            output_buf[i_bl, :, isubblock*nt:(isubblock+1)*nt] = 0
            continue
                
        for i_f in prange(nf):

            if reset_scales:
                s1[i_bl, i_f] = 0j
                s2[0, i_bl, i_f] = 0
                s2[1, i_bl, i_f] = 0
                N[i_bl, i_f] = 0

            if fixed_freq_weights[i_f] == 0:
                output_buf[i_bl, i_f, isubblock*nt:(isubblock+1)*nt] = 0
                continue

            cal_samp = cal_data[i_bl, i_f]
            for i_t in range(nt):
                isamp = input_data[i_bl, i_f, i_t]
                imask = input_tf_weights[i_f, i_t] == 0

                if imask:
                    output_buf[i_bl, i_f, isubblock*nt + i_t] = 0

                else:
                    #There is a bug in numba where I cannot set complex_var(128bit).real = real_var(64bit)
                    #See https://github.com/numba/numba/issues/3573

                    resulting_samp = isamp * cal_samp
                    N[i_bl, i_f] += 1
                    s1[i_bl, i_f] += resulting_samp
                    s2[0, i_bl, i_f] += resulting_samp.real**2
                    s2[1, i_bl, i_f] += resulting_samp.imag**2
                    
                    if not apply_rms and not sky_sub:
                        #We don't need to apply any kind of normalisation, so we can apply the cal right inside the first loop
                        #print(f"shapes of output_buf = {output_buf.shape}, cal_data = {cal_data.shape}")
                        output_buf[i_bl, i_f, isubblock*nt + i_t] = resulting_samp
            #pdb.set_trace()

    if apply_rms or sky_sub:

        #pdb.set_trace()

        if np.all(N == 0):
            return
    
        for i_bl in range(nbl):
            for i_f in range(nf):
                if N[i_bl, i_f] > 0:
                    means[i_bl, i_f] = s1[i_bl, i_f] / N[i_bl, i_f]
        
        if global_norm:
            global_N = N.sum()
            global_s2_real = s2[0].sum() + (N * means.real**2).sum() - 2 * (means.real * s1.real).sum()
            global_s2_imag = s2[1].sum() + (N * means.imag**2).sum() - 2 * (means.imag * s1.imag).sum()

            rms_real = np.sqrt(global_N * global_s2_real) / global_N
            rms_imag = np.sqrt(global_N * global_s2_imag) / global_N
            rms_val = np.sqrt(rms_real**2 + rms_imag**2) / np.sqrt(2)

            if rms_val == 0:
                multiplier = 1
            else:
                multiplier = target_rms_value / rms_val
            
        for i_bl in prange(nbl):
            if bl_weights[i_bl] == 0:
                continue
            for i_f in prange(nf):    
                if fixed_freq_weights[i_f] == 0:
                    continue            

                if not global_norm:
                    rms_real = np.sqrt(N[i_bl, i_f] * s2[0, i_bl, i_f] - s1[i_bl, i_f].real**2) / N[i_bl, i_f]
                    rms_imag = np.sqrt(N[i_bl, i_f] * s2[1, i_bl, i_f] - s1[i_bl, i_f].imag**2) / N[i_bl, i_f]
                    rms_val = np.sqrt(rms_real**2 + rms_imag**2) / np.sqrt(2)

                    if rms_val == 0:
                        multiplier = 1
                    else:
                        multiplier = target_rms_value / rms_val
            

                calval = cal_data[i_bl, i_f]
                #Loop through the data again and apply the various rescale factors and the cal soln
                for i_t in prange(nt):
                    isamp = input_data[i_bl, i_f, i_t]
                    imask = input_tf_weights[i_f, i_t] == 0

                    calibrated_samp = isamp * calval

                    if imask:
                        #Output has already been set to zero in the first loop. 
                        #We don't need to do anything here
                        pass
                    else:
                        if sky_sub and not apply_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (calibrated_samp -  means[i_bl, i_f])

                        elif sky_sub and apply_rms:
                            #output_buf[i_bl, i_f, isubblock*nt + i_t] = calibrated_samp
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (calibrated_samp - means[i_bl, i_f]) * multiplier

                        elif apply_rms and not sky_sub:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (calibrated_samp - means[i_bl, i_f]) * multiplier + means[i_bl, i_f] / calval
                
def create_tabs(vis_array, phasor_array, tab_array):
    '''
    vis_array - (nbl, nf, nt), Input visibilities
    phasor_array - (nsrc, nbl, nf), the phasor array to multiply with
    tab_array - (nsrc, nf, nt), the array that stores the output
    '''

    tab_array[:] = np.sum((vis_array[None, ...] * phasor_array[..., None]).real, axis=1)

@njit(parallel=False, cache=True)
def create_tabs_numba(vis_array, phasor_array, tab_array):
    '''
    vis_array - (nbl, nf, nt), Input visibilities
    phasor_array - (nsrc, nbl, nf), the phasor array to multiply with
    tab_array - (nsrc, nf, nt), the array that stores the output
    '''
    nbl, nf, nt = vis_array.shape
    nsrc, nbl, nf = phasor_array.shape

    for i_t in prange(nt):
        for isrc in prange(nsrc):
            for i_f in prange(nf):
                for i_bl in range(nbl):
                    tab_array[isrc, i_f, i_t] += (vis_array[i_bl, i_f, i_t] * phasor_array[isrc, i_bl, i_f]).real

class TAB_handler:
    
    def __init__(self, target_coords, plan, outdir):
        '''
        target_coords: np.ndarray/list - 1D (nsrc), contains a list of desired (RA, DEC) as astropy.SkyCoord objects
        plan: craco.pipeline_plan.PipelanPlan object
        outdir: str, path to the output directory where the tab filterbanks need to be saved
        '''
        self.target_coords = target_coords
        self.nsrc = len(self.target_coords)
        self.plan = plan
        self.outdir = outdir
        self.tab_array = np.zeros((self.nsrc, plan.nf, plan.nt), dtype=np.float32)
        self.initialise_filterbanks()
        #return self.fouts

    def create_phasors(self, phase_center_coord, uvws, freqs):
        '''
        Creates the phasors needed to phase up to a point source at a given coordinate
        Implements the np.exp(2*pi*j * f / c * blvec . dircos) component.

        Input
        -----
        phase_center_coord: astropy.SkyCoord, the coordinate of the phase center as astropy.skycoord object
        uvws: np.ndarray - 2D (nbl, 3), contains a list of UVW values (in seconds) for all baselines
        freqs: np.ndarray - 1D (nf), contains a list of all frequencies (in Hertz) for all channels 
        
        Returns
        -------
        phasor_array: np.ndarray - complex64 - 3D (nsrc, nbl, nf) - contains the phasors that can be multiplied directly with the visibilities
        '''

        nsrc = self.nsrc
        nbl = len(uvws)
        nf = len(freqs)
        fake_baseline_order = np.arange(nbl)
        phasor_array = np.zeros((nsrc, nbl, nf), dtype=np.complex64)

        for isrc, src_coord in enumerate(self.target_coords):
            lm = craco.coord2lm(src_coord, phase_center_coord)
            phasor_array[isrc] = craco.pointsource(1, lm, freqs, fake_baseline_order, uvws)

        self.phasor_array = phasor_array
        return self.phasor_array
    
    def initialise_filterbanks(self):
        common_hdr = {
                        'nbits':32,
                        'nchans': self.plan.nf,
                        'nifs': 1,
                        'tstart': self.plan.tstart.utc.mjd,
                        'tsamp': self.plan.tsamp_s.value,
                        'fch1': self.plan.fmin/1e6,
                        'foff': self.plan.foff/1e6,
        }
        self.fouts = []
        for isrc in range(self.nsrc):
            fname = os.path.join(self.outdir, f"tab_{isrc:02g}.fil")
            hdr = common_hdr.copy()
            hdr['src_raj_deg'] = self.target_coords[isrc].ra.deg
            hdr['src_dej_deg'] = self.target_coords[isrc].dec.deg
            self.fouts.append(sigproc.SigprocFile(fname, 'wb', hdr))

    def dump_to_fil(self):
        for isrc in range(self.nsrc):
             self.tab_array[isrc].T.tofile(self.fouts[isrc].fin)

    def __call__(self, vis_array):
        self.tab_array[:] = 0       #this is necessary, and only costs us ~80 micro-secs per block
        create_tabs_numba(vis_array, self.phasor_array, self.tab_array)
        self.dump_to_fil()
            
    def close(self):
        for isrc in range(self.nsrc):
            self.fouts[isrc].fin.close()


def calculate_num_good_cells(tf_weights, bl_weights, fixed_freq_weights):
    '''
    Counts how many good cells are there for a given block in total. Useful for keeping track of statistics
    Also returns num of good baselines and channels
    '''
    combined_tf_weights = tf_weights & fixed_freq_weights[:, None]
    tf_sum = combined_tf_weights.sum()
    bl_sum = bl_weights.sum()
    tot_sum = tf_sum * bl_sum
    return tot_sum, bl_sum

class FastPreprocess:

    #TODO -- add the capacity to write filterbank out for the masked values

    def __init__(self, blk_shape, cal_soln_array, values, fixed_freq_weights, beamid = 0, sky_sub = True, global_norm = True):
        self.cal_soln_array = self.make_averaged_cal_sol(cal_soln_array)
        self.dflag_nt = values.dflag_tblk
        self.dflag_fradius = values.dflag_fradius
        self.dflag_fthreshold = values.dflag_cas_threshold
        self.fixed_freq_weights = fixed_freq_weights
        assert len(fixed_freq_weights) == blk_shape[1]
        self.global_norm = global_norm
        self.target_input_rms = values.target_input_rms
        self.sky_sub = sky_sub
        self.beamid = beamid

        self.blk_shape = blk_shape
        self.num_fixed_good_chans = fixed_freq_weights.sum()
        self.total_num_cells = blk_shape[0] * blk_shape[1] * blk_shape[2]
        self.tf_num_cells = blk_shape[1] * blk_shape[2]
        self._initialise_internal_buffers()
        self._send_dummy_block()
        self._initialise_internal_buffers()

    def _initialise_internal_buffers(self):
        nbl, nf, nt = self.blk_shape
        self.interim_means = np.zeros((nbl, nf), dtype=np.complex128)
        self.s1 = np.zeros((nbl, nf), dtype = np.complex128)
        self.s2 = np.zeros((2, nbl, nf), dtype = np.float64)
        self.N = np.zeros((nbl, nf), dtype = np.int32)

        self.cas_block = np.zeros((nf, nt), dtype=np.float64)
        self.crs_block = np.zeros((nf, nt), dtype=np.float64)

        self.output_buf = np.zeros(self.blk_shape, dtype=np.complex64)

        self.num_good_nbl_pre = 0
        self.num_good_cells_pre = 0

        self.num_good_nbl_post = 0
        self.num_good_cells_post = 0

        self.num_dropped_cells_cumul = 0

        self.num_nblks = 0

        self.flagging_stats_fout = open(f"flagging_stats_log_b{self.beamid:02d}.csv", 'w')    
        self.flagging_stats_fout.write("#nblks\tnum_good_bl_pre_cumul\tnum_good_cells_pre_cumul\tnum_good_bl_post_cumul\tnum_good_cells_post_cumul\tnum_bad_cells_pre\tnum_bad_cells_post\texpected_block_shape\ttot_num_cells\tnum_fixed_good_chans\tdropped_packets_cumul\n")
        #self.output_buf = np.zeros((nrun, nuv, ncin, 2), dtype=np.int16)
        #self.lut = fast_bl2uv_mapping(nbl, nchan)       #nbl, nf, 3 - irun, iuv, ichan

    def _send_dummy_block(self):
        nbl, nf, nt = self.blk_shape
        dummy_block = np.zeros(self.blk_shape, dtype=np.complex64)
        dummy_input_tf_weights = np.ones((nf, nt), dtype=bool)
        dummy_bl_weights = np.ones(nbl, dtype=bool)
        self.__call__(dummy_block, dummy_bl_weights, dummy_input_tf_weights)

    def update_preflagging_statistics(self, tf_weights, bl_weights):
        #import pdb 
        #pdb.set_trace()
        #print(tf_weights.sum(), bl_weights.sum())
        num_good_cells, num_good_nbl = calculate_num_good_cells(tf_weights, bl_weights, self.fixed_freq_weights)
        self.num_bad_cells_pre_current = self.total_num_cells - num_good_cells
        self.num_good_cells_pre += num_good_cells
        self.num_good_nbl_pre += num_good_nbl
        self.num_dropped_cells_cumul += self.tf_num_cells - tf_weights.sum()

    def update_postflagging_statistics(self, tf_weights, bl_weights):
        num_good_cells, num_good_nbl = calculate_num_good_cells(tf_weights, bl_weights, self.fixed_freq_weights)
        self.num_bad_cells_post_current = self.total_num_cells - num_good_cells
        self.num_good_cells_post += num_good_cells
        self.num_good_nbl_post += num_good_nbl

    def log_flagging_stats(self):
        good_bls_pre, good_cells_pre = self.preflagging_stats
        good_bls_post, good_cells_post = self.postflagging_stats
        bad_cells_pre = self.num_bad_cells_pre_current
        bad_cells_post = self.num_bad_cells_post_current
        dropped_cells_cumul = self.num_dropped_cells_cumul / self.num_nblks

        out_str = f"{self.num_nblks:g}\t{good_bls_pre:.2f}\t{good_cells_pre:.2f}\t{good_bls_post:.2f}\t{good_cells_post:.2f}\t{bad_cells_pre:.2f}\t{bad_cells_post:.2f}\t{self.blk_shape}\t{self.total_num_cells:.2f}\t{self.num_fixed_good_chans:g}\t{dropped_cells_cumul:.2f}\n"
        self.flagging_stats_fout.write(out_str)
        self.flagging_stats_fout.flush()

    def close(self):
        if self.flagging_stats_fout is not None:
            self.flagging_stats_fout.write(f"#expected_blk_shape=({self.blk_shape}), num_fixed_good_chans = {self.fixed_freq_weights.sum()}\n")
            self.flagging_stats_fout.write("#num_good_bl_pre_cumul =  no of good baselines before flagging\n")
            self.flagging_stats_fout.write("#num_good_cells_pre_cumul =  no of good cells before flagging\n")
            self.flagging_stats_fout.write("#num_good_bl_post_cumul =  no of good baselines after flagging\n")
            self.flagging_stats_fout.write("#num_good_cells_post_cumul =  no of good cells after flagging\n")
            self.flagging_stats_fout.write("#All quantities are averaged by the no of blocks seen. So to get the true cumulative value, multiply by the corresponding nblks\n")
            self.flagging_stats_fout.close()
            
    @property
    def preflagging_stats(self):
        mean_cells = self.num_good_cells_pre / self.num_nblks
        mean_bls = self.num_good_nbl_pre / self.num_nblks
        return mean_bls, mean_cells
    
    @property
    def postflagging_stats(self):
        mean_cells = self.num_good_cells_post / self.num_nblks
        mean_bls = self.num_good_nbl_post / self.num_nblks
        return mean_bls, mean_cells

    @property
    def means(self):
        return self.s1 / self.N
    
    @property
    def stds(self):
        std_real_sq = (self.N * self.s2[0] - self.s1.real) 
        std_imag_sq = (self.N * self.s2[1] - self.s1.imag) 
        std = np.sqrt( (std_real_sq + std_imag_sq) / 2 ) / self.N
        return std
    
    @property
    def global_mean(self):
        return self.s1.sum() / self.N.sum()
    
    @property
    def global_std(self):
        global_N = self.N.sum()
        global_s2_real = self.s2[0].sum() + (self.N * self.means.real**2).sum() - 2 * (self.means.real * self.s1.real).sum()
        global_s2_imag = self.s2[1].sum() + (self.N * self.means.imag**2).sum() - 2 * (self.means.imag * self.s1.imag).sum()

        std_real_sq = global_N * global_s2_real
        std_imag_sq = global_N * global_s2_imag 
        std_val = np.sqrt( (std_real_sq + std_imag_sq) / 2 ) / global_N

        return std_val



    def make_averaged_cal_sol(self, cal_soln_array):
        '''
        Takes the calibration solution masked array, checks if the soln has the polarisation axis in it.
        If so, it takes the average across the pol axis and returns the data.
        Otherwise leaves it as is.
        It fills out the masks elements with 0s

        Input
        -----
        cal_soln_array: np.ma.core.MaskedArray - shape (nbl, nf, npol, ...)

        Returns
        -------
        cal_soln_array: np.array - shape (nbl, nf, ...)
        '''
        assert type(cal_soln_array) == np.ma.core.MaskedArray
        cal_soln_array.fill_value = 0
        if cal_soln_array.ndim == 4:
            return cal_soln_array.mean(axis=2).filled().squeeze()
        else:
            return cal_soln_array.filled().squeeze()

    def prepare_weights_and_block_from_masked_array(self, input_block):
        '''
        Extracts the bl_weights, fixed_freq_weights and input_tf_weights from a 3-D input_block
        Fills the masked values in the input_block with 0s

        Input
        -----
        input_block: np.ma.core.MaskedArray
                    The input data block of shape (nbl, nf, nt) with an associated mask

        Returns
        -------
        input_data: np.ndarray
                    The input data block of shape (nbl, nf, nt) after filling out masked elements with 0s
        bl_weights: np.ndarray
                    The baseline wights array of shape (nbl)
        input_tf_weights: np.ndarray
                    The input weights array of shape (nf, nt)
        fixed_freq_weights: np.ndarray
                    The fixed frequency weights array of shape (nf)
        '''
        masks = input_block.mask
        bl_weights = np.empty(masks.shape[0],dtype=bool)
        for ii, bl in enumerate(masks):
            bl_weights[ii] = ~np.all(bl)

        tf_weights = ~input_block[bl_weights].mask[0]

        input_block.fill_value = 0
        return input_block.filled().squeeze(), bl_weights, tf_weights


    def __call__(self, input_block, bl_weights = None, input_tf_weights = None):
        '''
        Performs the preprocessing steps on input_block
        '''
        if type(input_block) == np.ma.core.MaskedArray:
            input_data, bl_weights, input_tf_weights = self.prepare_weights_and_block_from_masked_array(input_block)
        else:
            input_data = input_block
            if bl_weights is None or input_tf_weights is None:
                raise ValueError(f"I need bl_weights, fixed_freq_weights and input_tf_weights to be provided if the input block is not a masked array")
            assert len(bl_weights) == self.blk_shape[0]
            assert input_tf_weights.shape == self.blk_shape[1:]
    
        self.update_preflagging_statistics(input_tf_weights, bl_weights)

        get_phase_varying_dynamic_rfi_masks(input_block,
                                            finest_nt = self.dflag_nt,
                                            bl_weights = bl_weights,
                                            tf_weights = input_tf_weights,
                                            freq_radius= self.dflag_fradius,
                                            freq_threshold = self.dflag_fthreshold)

        self.update_postflagging_statistics(input_tf_weights, bl_weights)
        self.num_nblks += 1

        self.log_flagging_stats()

        fast_preprocess_sos(input_data=input_data,
                            bl_weights=bl_weights,
                            fixed_freq_weights=self.fixed_freq_weights,
                            input_tf_weights=input_tf_weights,
                            output_buf=self.output_buf,
                            isubblock=0,
                            means=self.interim_means,
                            s1 = self.s1,
                            s2 = self.s2,
                            N=self.N,
                            calsoln_data=self.cal_soln_array,
                            target_input_rms=self.target_input_rms,
                            sky_sub=self.sky_sub,
                            global_norm=self.global_norm)



class Calibrate:
    #TODO remove the dependence of Calibrate on plan -- needs to propage through to the calibration.py's CalSolution class
    def __init__(self, plan, block_dtype, miriad_gains_file, baseline_order, keep_masks = True):
        self.gains_file = miriad_gains_file
        self.baseline_order = baseline_order
        self.plan = plan
        self.reload_gains()

        if block_dtype not in [np.ndarray, np.ma.core.MaskedArray, dict]:
            raise ValueError("Unknown dtype of block provided")

        #self.block_dtype = block_dtype
        self.keep_masks = keep_masks
        

    def reload_gains(self):
        #self.ant_gains, _ = calibration.load_gains(self.gains_file)
        #self.gains_array = calibration.soln2array(self.ant_gains, self.baseline_order)
        self.plan.values.calibration = self.gains_file
        calsoln_obj = calibration.CalibrationSolution(plan = self.plan)
        self.gains_array = calsoln_obj.solarray.copy()
        self.gains_pol_avged_array = self.gains_array.mean(axis=-2, keepdims=True).astype(self.gains_array.dtype)
        if type(self.gains_array) == np.ma.core.MaskedArray:
            self.sol_isMasked = True
        else:
            self.sol_isMasked = False

    def apply_calibration(self, block):
        block_isMasked, nPol, block_type = get_isMasked_nPol(block)
        #assert block_type == self.block_dtype, f"You Liar!, {block_type}, {self.block_dtype}"

        if block_type == dict:
            #print(f"Baseline order is: {self.baseline_order}")
            for ibl, blid in enumerate(self.baseline_order):
                #print(f"ibl {ibl}, blid={blid}, npol= {nPol}, bldata.shape = {block[blid].shape}, gains_array.shape = {self.gains_array.shape, self.gains_pol_avged_array.shape}")
                if nPol == 2:
                    block[blid] = self.gains_array[ibl, ...] * block[blid] 
                elif nPol ==1:
                    block[blid] =  self.gains_pol_avged_array[ibl, ...] * block[blid]
                elif nPol ==0:
                    block[blid] = self.gains_pol_avged_array[ibl, ...].squeeze()[..., None] * block[blid]
                else:
                    raise ValueError(f"Expected nPol 0, 1, or 2, but got {nPol}")
                
                #print(f"~~~~~~~~~~ 'apply_calibration() says' The shape of block[{blid}] is {block[blid].shape}")
                
                if self.keep_masks or (not block_isMasked and not self.sol_isMasked):
                    pass
                else:
                    block[blid] = np.asarray(block[blid])
            return block

        else:
            
            if nPol == 2:
                calibrated_block = self.gains_array * block
            elif nPol == 1:
                calibrated_block = self.gains_pol_avged_array * block
            elif nPol == 0:
                calibrated_block = self.gains_pol_avged_array.squeeze()[..., None] * block
            else:
                raise ValueError(f"Expected nPol 0, 1, or 2, but got {nPol}")

            if self.keep_masks or (not block_isMasked and not self.sol_isMasked):
                return calibrated_block
            else:
                calibrated_block.data[calibrated_block.mask] = 0
                return np.asarray(calibrated_block)


class RFI_cleaner:
    def __init__(self, block_dtype, baseline_order):
        '''
        block_dtype: dtype (np.ndarry or np.ma.core.MaskedArray or dict)
                    The data type of the block that will be passed
        baseline_order: list or 1-D numpy array
                A list of the blids in the same order as they would
                be arranged in the block if it is an array. Not required 
                if block_dtype is a dict (where the block itself contains
                the blids as keys)
        '''
        if block_dtype not in [np.ndarray, np.ma.core.MaskedArray, dict]:
            raise ValueError("Only np.ndarrays and np.ma.core.MaskedArrays are currently supported")

        self.block_dtype = block_dtype
        self.baseline_order = baseline_order

    def flag_chans(self, block, chanrange, flagval=0):
        '''
        Sets a given range of channels to value flagval
        '''
        self.isMasked, self.pol_axis_exists, _ = get_isMasked_nPol(block)
        for ibl, bldata in enumerate(block):
            if self.isMasked:
                block[ibl].mask[chanrange, ...] = True
                block[ibl][chanrange, ...] = flagval
            else:
                block[ibl][chanrange, ...] = flagval
        return block

    def get_freq_mask(self, baseline_data, threshold=5.0):
        #Take the mean along the pol axis and rms along the time axis

        if self.pol_axis_exists:
            rms = baseline_data.mean(axis=1).std(axis=-1).squeeze()
        else:
            rms = baseline_data.std(axis=-1).squeeze()

        mask, votes = iqrm_mask(rms, radius = len(rms), threshold=threshold)
        return mask

    def get_time_mask(self, baseline_data, threshold = 5.0):
        #Take the mean along the pol axis and rms along freq axis
        if self.pol_axis_exists:
            #print("Shape of baseline_data given to me is=  ",baseline_data.shape)
            rms = baseline_data.mean(axis=1).std(axis=0).squeeze()
        else:
            #print("Shape of baseline_data given to me is=  ~~~~~~",baseline_data.shape)
            rms = baseline_data.std(axis=0).squeeze()

        #print("LEN(RMS)  =  ", len(rms))
        mask, votes = iqrm_mask(rms, radius = len(rms)/ 10, threshold=threshold)
        return mask


    def get_IQRM_autocorr_masks(self, block, freq=True, time=True):
        autocorr_masks = {}
        for ibl, baseline_data in enumerate(block):
            a1, a2 = bl2ant(self.baseline_order[ibl])
            if a1 != a2:
                continue
            
            if freq:
                autocorr_freq_mask = self.get_freq_mask(baseline_data, threshold=5)
                autocorr_masks[str(a1) + 'f'] = autocorr_freq_mask 
            if time:
                autocorr_time_mask = self.get_time_mask(baseline_data)
                autocorr_masks[str(a1) + 't'] = autocorr_time_mask
        return autocorr_masks

    def clean_bl_using_autocorr_mask(self, ibl, baseline_data, autocorr_masks, freq, time):
        #print("Cleaning autocorr for ibl", ibl, "the ant is going to be ", bl2ant(self.baseline_order[ibl]))
        if len(autocorr_masks) == 0:
            return baseline_data
        ant1, ant2 = bl2ant(self.baseline_order[ibl])
        if freq:
            autocorr_freq_mask = autocorr_masks[str(ant1) + 'f'] | autocorr_masks[str(ant2) + 'f']
            if self.isMasked:
                baseline_data.mask[autocorr_freq_mask] = True
                baseline_data.data[autocorr_freq_mask] = 0
            else:
                baseline_data[autocorr_freq_mask] = 0

        if time:
            autocorr_time_mask = autocorr_masks[str(ant1) + 't'] | autocorr_masks[str(ant2) + 't']
            if self.isMasked:
                baseline_data.mask[..., autocorr_time_mask] = True
                baseline_data.data[..., autocorr_time_mask] = 0
            else:
                baseline_data[..., autocorr_time_mask] = 0

        return baseline_data


    def run_IQRM_cleaning(self, block, maf, mat, mcf, mct, mcasf, mcast):
        '''
        Does the IQRM magic

        block needs to be a np.ndarray or np.ma.core.MaskedArray
        Each block data should have shape (nbl, nf, npol, nt) or (nbl, nf, nt)

        All of the remaining flags need boolean values (True/False)
        maf: Mask auto-corrs in freq (True to enable, False to disable)
        mat: Mask auto-corrs in time (True to enable, False to disable)
        mcf: Mask cross-corrs in freq (True to enable, False to disable)
        mct: Mask cross-corrs in time (True to enable, False to disable)
        mcasf: Mask cross-amp sum in freq (True to enable, False to disable)
        mcast: Mask cross-amp sum in time (True to enable, False to disable)

        Returns
        -------
        autocorr_masks: dict
            Dictionary containing autocorr_masks keyed by antid (1 - 36) + 'f' or 't', valued by a 1-D numpy array of len nf/nt
        crosscorr_masks: dict
            Dictionary containing crosscorr_masks keyed by blid (256*a1 + a2) + 'f' or 't', valued by 1-D numpy array of len nf/nt
        cas_masks: dict
            Single element dictionary keyed by 'f' or 't', valued by a 1-D numpy array of len nf/nt
        '''
        autocorr_masks = {}
        crosscorr_masks = {}
        cas_masks = {}

        assert type(block) == self.block_dtype, "You Liar!"

        self.isMasked, self.pol_axis_exists, _ = get_isMasked_nPol(block)

        if maf or mat:
            autocorr_masks = self.get_IQRM_autocorr_masks(np.abs(block), maf, mat)

            #print("autocorr masks are", autocorr_masks)
        
        if mcast or mcasf:
            cas_sum = np.zeros_like(block[0])

        if maf or mat or mcf or mct or mcasf or mcast:
           for ibl, baseline_data in enumerate(block):

                if self.isMasked:
                    block[ibl].data[baseline_data.mask] = 0  #Zero the data where the data is already flagged, to avoid the IQRM getting too swayed by the bad samples
                
                if maf or mat:
                   block[ibl] = self.clean_bl_using_autocorr_mask(ibl, np.abs(baseline_data), autocorr_masks, freq=maf, time=mat)


                if mcf or mct or mcasf or mcast:

                    ant1, ant2 = bl2ant(self.baseline_order[ibl])
                    if ant1 == ant2:            
                       continue

                    if mct:
                        #print("Shape of baseline_data = ", baseline_data.shape)
                        bl_time_mask = self.get_time_mask(np.abs(baseline_data))
                        #print("Shape of time_mask = ", bl_time_mask.shape)

                        if self.isMasked:
                            baseline_data.data[..., bl_time_mask] = 0
                            baseline_data.mask[..., bl_time_mask] = True
                        else:
                            baseline_data[..., bl_time_mask] = 0

                        crosscorr_masks[str(ibl) + 't'] = bl_time_mask

                    if mcf:
                        bl_freq_mask = self.get_freq_mask(np.abs(baseline_data), threshold=5.0)
                        if self.isMasked:
                            baseline_data.data[bl_freq_mask, ...] = 0
                            baseline_data.mask[bl_freq_mask, ...] = True
                        else:
                            baseline_data[bl_freq_mask, ...] = 0

                        crosscorr_masks[str(ibl) + 'f'] = bl_freq_mask

                    block[ibl] = baseline_data
                    if mcasf or mcast:
                        cas_sum += np.abs(baseline_data)


        if mcasf or mcast:
            #Finally find bad samples in the CAS
            cas_masks['f'] = self.get_freq_mask(np.abs(cas_sum), threshold=5)  
            if self.isMasked:
                cas_sum.data[cas_masks['f']] = 0
                cas_sum.mask[cas_masks['f']] = True
            else:
                cas_sum[cas_masks['f']] = 0


            cas_masks['t'] = self.get_time_mask(np.abs(cas_sum))
            #We don't bother zero-ing the cas-sum now since it will not be used any further
            #if self.isMasked:
                #cas_sum[..., cas_masks['t']].data = 0
                #cas_sum[..., cas_masks['t']].mask = True
            #else:
                #cas_sum[..., cas_masks['t']] = 0

            if self.isMasked:
                block.data[:, cas_masks['f'], ...] = 0
                block.mask[:, cas_masks['f'], ...] = True

                block.data[..., cas_masks['t']] = 0
                block.mask[..., cas_masks['t']] = True

            else:
                block[:, cas_masks['f'], ...] = 0
                block[..., cas_masks['t']] = 0

        return block, autocorr_masks, crosscorr_masks, cas_masks

def fill_masked_values(block, fill_value = None):
    '''
    Fill the masked elements with their fill_values, or the provided fill_value
    If the there is no mask, then the same block is returned
    '''
    if type(block) == dict:
        for ibl, bldata in block.items():
            if type(bldata) == np.ma.core.MaskedArray:
                if fill_value is not None:
                    block.fill_value = fill_value
                block[ibl] = bldata.filled()
            else:
                pass
        return block

    elif type(block) == np.ndarray:
        return block

    elif type(block) == np.ma.core.MaskedArray:
        if fill_value is not None:
            block.fill_value = fill_value
        return block.filled()


def get_dm_delays(dm_samps, freqs):
    '''
    Returns delays for each freq in freqs array
    dm_samps is the total delay across the whole band
    delay for fmin will be zero
    '''
    fmin = np.min(freqs)
    fmax = np.max(freqs)
    delay = dm_samps * (1 / fmin**2 - 1 / freqs**2)  / (1 / fmin**2 - 1 / fmax**2)
    delay_samps = np.round(delay).astype(int)
    return delay_samps

def get_dm_pccc(freqs, dm_samps, tsamp):
    '''
    freqs in Hz
    tsamp in s
    '''
    delay_s = dm_samps * tsamp
    fmax = np.max(freqs) * 1e-9
    fmin = np.min(freqs) * 1e-9
    dm_pccc = delay_s / 4.15 / 1e-3 / (1 / fmin**2 - 1 / fmax**2)
    return dm_pccc

def get_dm_samps(freqs, dm_pccc, tsamp):
    '''
    freqs in Hz
    tsamp in s
    '''
    fmax = np.max(freqs) * 1e-9
    fmin = np.min(freqs) * 1e-9
    delays_s = 4.15 * 1e-3 * dm_pccc * (1 / fmin**2 - 1 / fmax**2)
    
    delays_samps = np.round(delays_s / tsamp).astype(int)
    dm_samps = delays_samps
    return dm_samps

class Dedisp:
    def __init__(self, freqs, tsamp,  dm_samps = None, dm_pccc = None):
        self.fch1 = freqs[0]
        self.foff = np.abs(freqs[1] - freqs[0])
        self.freqs = freqs
        self.nchans = len(self.freqs)
        if dm_samps is None:
            if dm_pccc is None:
                raise ValueError("You need to specify either dm_samps or dm_pccc")
            dm_samps = get_dm_samps(freqs, dm_pccc, tsamp)

        self.delays_samps = get_dm_delays(dm_samps, self.freqs)
        #print(f"The computed delays_samps are: {self.delays_samps}")
        self.dm = dm_samps
        self.dm_pccc = get_dm_pccc(freqs, dm_samps, tsamp)
        self.dm_history = None

    def dedisperse(self, iblock, inblock):
        if type(inblock) in [np.ndarray, np.ma.core.MaskedArray]:       #I removed the support for numpy.maksed_array on 12 Feb 2024, but I should add it back"
            block = inblock
        else:
            raise TypeError(f"Expected either np.ndarray or np.ma.core.MaskedArray, but got {type(inblock)}")

        if iblock == 0:
            history_shape = list(block.shape)
            history_shape[-1] = self.dm
            history_shape = tuple(history_shape)

            self.dm_history = np.zeros(history_shape, dtype=block.dtype)
       
        if self.dm == 0:
            return inblock

        attached_block = np.concatenate([self.dm_history, block], axis=-1)
        rolled_block = np.zeros_like(attached_block)
        for ichan in range(self.nchans):
            rolled_block[:, ichan, ...] = np.roll(attached_block[:, ichan, ...], self.delays_samps[ichan])

        self.dm_history = attached_block[..., -self.dm:]

        return rolled_block[..., self.dm:]
        
