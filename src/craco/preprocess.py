from . import calibration
from iqrm import iqrm_mask
import numpy as np
from craft.craco import bl2ant, bl2array


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
            new_block[ibl] = (bldata - np.median(bldata, axis=-1, keepdims=True)) * (target_input_rms / existing_rms)[..., None]
            #print(f"====>> The shape of normalised block[ibl] for ibl{ibl} is {block[ibl].shape}")
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        existing_rms = block.std(axis=-1) / np.sqrt(2)   
        new_block = (block - np.median(block, axis=-1, keepdims = True)) * (target_input_rms / existing_rms)[..., None]
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
    if nPol == 0:
        print("No pol axis to scrunch, returning the block as it is")
        return block
    if type(block) == dict:
        for ibl, bldata in block.items():
            assert bldata.ndim == 3, f"Exptected 3 dimensions (nf, npol, nt), but got {bldata.ndim}, shape={bldata.shape}"#, block = {block},\n bldata= {bldata}"
            block[ibl] = bldata.mean(axis=1, keepdims=keepdims)
            if not keepdims:
                block[ibl] = block[ibl].squeeze()
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        assert block.ndim == 4, f"Expected 4 dimensions (nbl, nf, npol, nt), but got {bldata.ndim}"
        block = block.mean(axis=2, keepdims=keepdims)
        if not keepdims:
            block[ibl] = block[ibl].squeeze()
    return block



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
        self.gains_pol_avged_array = self.gains_array.mean(axis=-2, keepdims=True)
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

    def get_freq_mask(self, baseline_data, threshold=3.0):
        #Take the mean along the pol axis and rms along the time axis

        if self.pol_axis_exists:
            rms = baseline_data.mean(axis=1).std(axis=-1).squeeze()
        else:
            rms = baseline_data.std(axis=-1).squeeze()

        mask, votes = iqrm_mask(rms, radius = len(rms) / 10, threshold=threshold)
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
                autocorr_freq_mask = self.get_freq_mask(baseline_data)
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
        Each block data should have shape (nbl, nf, npol, nt) or (nbl, nf, nt) of absolute values (np.abs(block))

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
            autocorr_masks = self.get_IQRM_autocorr_masks(block, maf, mat)

            #print("autocorr masks are", autocorr_masks)
        
        if mcast or mcasf:
            cas_sum = np.zeros_like(block[0])

        if maf or mat or mcf or mct or mcasf or mcast:
           for ibl, baseline_data in enumerate(block):

                if self.isMasked:
                    block[ibl].data[baseline_data.mask] = 0  #Zero the data where the data is already flagged, to avoid the IQRM getting too swayed by the bad samples
                
                if maf or mat:
                   block[ibl] = self.clean_bl_using_autocorr_mask(ibl, baseline_data, autocorr_masks, freq=maf, time=mat)


                if mcf or mct or mcasf or mcast:

                    ant1, ant2 = bl2ant(self.baseline_order[ibl])
                    if ant1 == ant2:            
                       continue

                    if mct:
                        #print("Shape of baseline_data = ", baseline_data.shape)
                        bl_time_mask = self.get_time_mask(baseline_data)
                        #print("Shape of time_mask = ", bl_time_mask.shape)

                        if self.isMasked:
                            baseline_data.data[..., bl_time_mask] = 0
                            baseline_data.mask[..., bl_time_mask] = True
                        else:
                            baseline_data[..., bl_time_mask] = 0

                        crosscorr_masks[str(ibl) + 't'] = bl_time_mask

                    if mcf:
                        bl_freq_mask = self.get_freq_mask(baseline_data)
                        if self.isMasked:
                            baseline_data.data[bl_freq_mask, ...] = 0
                            baseline_data.mask[bl_freq_mask, ...] = True
                        else:
                            baseline_data[bl_freq_mask, ...] = 0

                        crosscorr_masks[str(ibl) + 'f'] = bl_freq_mask

                    block[ibl] = baseline_data
                    cas_sum += baseline_data


        if mcasf or mcast:
            #Finally find bad samples in the CAS
            cas_masks['f'] = self.get_freq_mask(cas_sum)  
            if self.isMasked:
                cas_sum.data[cas_masks['f']] = 0
                cas_sum.mask[cas_masks['f']] = True
            else:
                cas_sum[cas_masks['f']] = 0


            cas_masks['t'] = self.get_time_mask(cas_sum)
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



def get_dm_delays(dm_samps, freqs):
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
    #dm_samps = np.max(delays_samps) - np.min(delays_samps)
    dm_samps = delays_samps
    #print(f"Delays_s are", delays_s, delays_samps, dm_samps)
    return dm_samps

class Dedisp:
    def __init__(self, freqs, tsamp, baseline_order, dm_samps = None, dm_pccc = None):
        self.fch1 = freqs[0]
        self.foff = np.abs(freqs[1] - freqs[0])
        self.freqs = freqs
        self.nchans = len(self.freqs)
        self.baseline_order = baseline_order
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

        if type(inblock) == dict:
            block = bl2array(inblock)
        elif type(inblock) in [np.ndarray, np.ma.core.MaskedArray]:
            block = inblock
        else:
            raise TypeError(f"Expected either np.ndarray or np.ma.core.MaskedArray or dict, but got {type(block)}")

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
        #import IPython
        
        if type(inblock) == dict:
            for ibl, blid in enumerate(self.baseline_order):
                #print(ibl,blid, inblock[blid].shape, attached_block[ibl, ..., self.dm:].shape)
                #print(inblock[blid].mask)
                inblock[blid] = rolled_block[ibl, ..., self.dm:]
                #print(f"Type of the output vis array is -- {type(inblock[blid])}")
                #print(inblock[blid].mask)
            #IPython.embed()
            return inblock

        return rolled_block[..., self.dm:]
        
