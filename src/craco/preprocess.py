from . import calibration
from iqrm import iqrm_mask
import numpy as np
from craft.craco import bl2ant, bl2array
from numba import njit

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

    #import IPython
    #IPython.embed()
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

def fast_preprpocess(input_block, output_buf, output_mask, isubblock, Ai, Qi, N, calsoln, target_input_rms, sky_sub = False, reset_scales = False, input_mask = None):
    '''
    Loops over all dimensions of the input_block. Applies the calibration soln,
    Measures the input levels, and optionally rescales and does the sky subtraction.

    input_block: Input data array - (nbl, nf, nt). Can be a masked array or np.ndarray - complex64
    input_mask: Input mask array - (nbl, nf, nt). If not provided, input_block needs to be a masked array
    output_buf: Output data array - (nbl, nf, nt*nsubblock). Won't be a masked array - complex64
    output_mask: Output masks array after combining the masks produced by the dynamic flagger
    isubblock: integer - output_buf can hold mulitple input_blocks - isubblock is the counter
    Ai: numpy array to keep track of the running mean (nbl, nf) - complex64
    Qi: numpy array to keep track of the running rms (nbl, nf) - complex64
    N: numpy array to keep track of the nnumbers added so far (nbl, nf) - int16
        This needs to be initialised with ones at the very beginning, otherwise bad things will happen
    calsoln: numpy array containing the calibration solution (nbl, nf) - complex64
    target_input_rms: float, the rms value which needs to be set. If None, data will not be rescaled
    sky_sub: bool, Whether to do mean subtraction or not.
    reset_scales: bool, whether to keep on accumulating Ai and Qi values across blocks, or reset to zero at the start of this block
                    Note -- reset_scales flag should set to true when this function is called for the first time, or we need to ensure that
                    Ai, Qi and N are initialised properly outside this function.
    '''
    #TODO -- think about whether changing the order of the input_block could speed things up a bit more
    #TODO -- include the computation of CAS and the masks inside this function's first pass over the data
    #        Right now it is a bit too complicated to allow for a different nt for the flagger
    nbl, nf, nt = input_block.shape

    if input_mask is None:
        assert type(input_block)==np.ma.core.MaskedArray, f"Given - {type(input_block)}"
        input_data = input_block.data
        #input_mask = np.any(input_block.mask, axis=0)   #We take the OR of all mask values along the Basline axis given that Keith says that if a packet is dropped, all baselines are gauranteed to be flagged for that time-freq spaxel
        input_mask = input_block.mask

    else:
        input_mask = input_mask
        assert input_mask.shape == input_block.shape, f"Shape of the input masks should match that of input_block - Input_block.shape = {input_block.shape}, Given shape = {input_mask.shape}"

        if type(input_block) == np.ma.core.MaskedArray:
            input_data = input_block.data
        elif type(input_block) == np.ndarray:
            input_data = input_block
        else:
            raise ValueError(f"dtype of input block should be np.ma.core.MaskedArray, or nd.ndarray, found = {type(input_block)}")

    assert type(calsoln)==np.ma.core.MaskedArray, f"Given - {type(calsoln)}"
 
    cal_data = calsoln.data
    cal_mask = calsoln.mask

    for i_bl in range(nbl):
        for i_f in range(nf):
        
            if reset_scales:
                Qi[i_bl, i_f] = 0
                zeroth_samp = input_data[i_bl, i_f, 0]
                zeroth_samp_mask = input_mask[i_bl, i_f, 0]

                if not zeroth_samp_mask:
                    Ai[i_bl, i_f] = zeroth_samp
                    N[i_bl, i_f] = 2
                else:
                    Ai[i_bl, i_f] = 0
                    N[i_bl, i_f] = 1

            ical_mask = cal_mask[i_bl, i_f]
            if ical_mask:
                #Skip all the computation, directly set the output values to zero, and go to the next channel
                output_buf[i_bl, i_f, isubblock*nt:(isubblock+1)*nt] = 0
                Ai[i_bl, i_f] = 0
                Qi[i_bl, i_f] = 0
                N[i_bl, i_f] = 1
                continue
            
            for i_t in range(nt):
                isamp = input_data[i_bl, i_f, i_t]
                imask = input_mask[i_bl, i_f, i_t]

                if imask:
                    output_buf[i_bl, i_f, isubblock*nt + i_t] = 0
                    continue
                else:
                    Qi[i_bl, i_f] = Qi[i_bl, i_f] + (N[i_bl, i_f] -1)/ N[i_bl, i_f] * (isamp - Ai[i_bl, i_f])**2
                    Ai[i_bl, i_f] = Ai[i_bl, i_f] + (isamp - Ai[i_bl, i_f])/ N[i_bl, i_f]
                    N[i_bl, i_f] += 1

                    if target_input_rms is None and not sky_sub:
                        #We don't need to apply any kind of normalisation, so we can apply the cal right inside the first loop
                        #print(f"shapes of output_buf = {output_buf.shape}, cal_data = {cal_data.shape}")
                        output_buf[i_bl, i_f, isubblock*nt + i_t] = isamp * cal_data[i_bl, i_f]

            if target_input_rms or sky_sub:
                rms_val_real = np.sqrt(Qi[i_bl, i_f].real / N[i_bl, i_f].real)
                rms_val_imag = np.sqrt(Qi[i_bl, i_f].imag / N[i_bl, i_f].imag)
                rms_val = np.sqrt(rms_val_real**2 + rms_val_imag**2) / np.sqrt(2)

                multiplier = cal_data[i_bl, i_f] * target_input_rms / rms_val
                #Loop through the data again and apply the various rescale factors and the cal soln
                for i_t in range(nt):
                    isamp = input_data[i_bl, i_f, i_t]
                    imask = input_mask[i_bl, i_f, i_t]

                    if imask:
                        #Output has already been set to zero in the first loop. 
                        #We don't need to do anything here
                        pass
                    else:
                        if sky_sub and not target_input_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = isamp - Ai[i_bl, i_f]

                        elif sky_sub and target_input_rms:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp - Ai[i_bl, i_f]) * multiplier

                        elif target_input_rms and not sky_sub:
                            output_buf[i_bl, i_f, isubblock*nt + i_t] = (isamp - Ai[i_bl, i_f]) * multiplier + Ai[i_bl, i_f]


                

                


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
        if type(inblock) in [np.ndarray, np.ma.core.MaskedArray]:
            block = inblock
        else:
            raise TypeError(f"Expected either np.ndarray or np.ma.core.MaskedArray, but got {type(block)}")

        if iblock == 0:
            history_shape = list(block.shape)
            history_shape[-1] = self.dm
            history_shape = tuple(history_shape)

            self.dm_history = np.zeros(history_shape, dtype=block.dtype)

        attached_block = np.concatenate([self.dm_history, block], axis=-1)
        rolled_block = np.zeros_like(attached_block)
        for ichan in range(self.nchans):
            rolled_block[:, ichan, ...] = np.roll(attached_block[:, ichan, ...], self.delays_samps[ichan])

        self.dm_history = attached_block[..., -self.dm:]

        return rolled_block[..., self.dm:]
        
