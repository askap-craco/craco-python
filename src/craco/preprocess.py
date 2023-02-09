from . import calibration
from iqrm import iqrm_mask
import numpy as np
from craft.craco import bl2ant

def normalise(block, mean = 0, std = 1):
    '''
    Normalises each baseline and channel in the visibility block to the
    specified mean and std values along the time axis.
    It the array/dictionary in place

    Input
    -----
    block: np.ndarray or np.ma.masked_array or dict
            Input visbility data of shape (nbl, nf, [npol], nt) if array, or,
            Visibility dict with nbl arrays/masked arrays of 
            (nf, [npol], nt) shape each. The npol axis is optional
    mean: float
            Desired mean of each channel
    std: float
            Desired std of each channel
    '''
    if type(block) == dict:
        for ibl, bldata in block.items():
            existing_rms = bldata.std(axis=-1, keepdims=True)
            block[ibl] = (bldata - bldata.mean(axis=-1, keepdims=True)) * (std / existing_rms) + mean
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        existing_rms = block.std(axis=-1, keepdims=True)
        block = (block - block.mean(axis=-1, keepdims = True)) * (std / existing_rms) + mean
    else:
        raise Exception("Unknown type of block provided - expecting dict, np.ndarray, or np.ma.core.MaskedArray")
    return block


def average_pols(block):
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
    if type(block) == dict:
        for ibl, bldata in block.items():
            assert bldata.ndim == 3, f"Exptected 3 dimensions (nf, npol, nt), but got {bldata.ndim}"
            block[ibl] = bldata.mean(axis=1).squeeze()
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        assert block.ndim == 4, f"Expected 4 dimensions (nbl, nf, npol, nt), but got {bldata.ndim}"
        block = block.mean(axis=2).squeeze()
    return block


def get_isMasked_nPol(block):
    data = block
    if type(data) == np.ndarray:
        isMasked = False
        ndim = data.ndim
    elif type(data) == np.ma.core.MaskedArray:
        isMasked = True
        ndim = data.ndim

    if ndim == 3:
        nPol = 0
    elif ndim == 4:
        nPol = data.shape[-2]
    else:
        raise Exception(f"ndim of a single baseline can only be 2 or 3, found {ndim}")
    return isMasked, nPol



class Calibrate:
    def __init__(self, block_dtype, miriad_gains_file, baseline_order, keep_masks = True):
        self.gains_file = miriad_gains_file
        self.reload_gains()

        if block_dtype not in [np.ndarray, np.ma.core.MaskedArray]:
            raise ValueError("Unknown dtype of block provided")

        self.block_dtype = block_dtype
        self.keep_masks = keep_masks
        self.baseline_order = baseline_order

    def reload_gains(self):
        self.ant_gains = calibration.load_gains(self.gains_file)
        self.gains_array = calibration.soln2array(self.ant_gains, self.baseline_order)
        self.gains_pol_avged_array = self.gains_array.mean(axis=-2, keepdims=True)
        if type(self.gains_array) == np.ma.core.MaskedArray:
            self.sol_isMasked = True
        else:
            self.sol_isMasked = False

    def apply_calibration(self, block):
        block_isMasked, nPol = get_isMasked_nPol(block)

        if nPol == 2:
            calibrated_block = self.gains_array * block
        if nPol == 1:
            calibrated_block = self.gains_pol_avged_array * block
        if nPol == 0:
            calibrated_block = self.gains_pol_avged_array.squeeze() * block

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
        if block_dtype not in [np.ndarray, np.ma.core.MaskedArray]:
            raise ValueError("Only np.ndarrays and np.ma.core.MaskedArrays are currently supported")

        self.block_dtype = block_dtype
        self.baseline_order = baseline_order

    def get_freq_mask(self, baseline_data, threshold=3.0):
        #Take the mean along the pol axis and rms along the time axis

        if self.pol_axis_exists:
            rms = baseline_data.mean(axis=1).std(axis=-1)    
        else:
            rms = baseline_data.std(axis=-1)

        mask, votes = iqrm_mask(rms, radius = len(rms) / 10, threshold=threshold)
        return mask

    def get_time_mask(self, baseline_data, threshold = 5.0):
        #Take the mean along the pol axis and rms along freq axis
        if self.pol_axis_exists:
            rms = baseline_data.mean(axis=1).std(axis=0)
        else:
            rms = baseline_data.std(axis=0)

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
        ant1, ant2 = bl2ant(self.baseline_order[ibl])
        if freq:
            autocorr_freq_mask = autocorr_masks[str(ant1) + 'f'] | autocorr_masks[str(ant2) + 'f']
            if self.isMasked:
                baseline_data[autocorr_freq_mask].mask = True
                baseline_data[autocorr_freq_mask].data = 0
            else:
                baseline_data[autocorr_freq_mask] = 0

        if time:
            autocorr_time_mask = autocorr_masks[str(ant1) + 't'] | autocorr_masks[str(ant2) + 't']
            if self.isMasked:
                baseline_data[..., autocorr_time_mask].mask = True
                baseline_data[..., autocorr_time_mask].data = 0
            else:
                baseline_data[..., autocorr_time_mask] = 0

        return baseline_data


    def run_IQRM_cleaning(self, block, maf, mat, mcf, mct, mcasf, mcast):
        '''
        Does the IQRM magic

        block needs to be a np.ndarray or np.ma.core.MaskedArray
        Each block data should have shape (nbl, nf, npol, nt) or (nbl, nf, nt)

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

        self.isMasked, self.pol_axis_exists = get_isMasked_nPol(block[0])

        if maf or mat:
            autocorr_masks = self.get_IQRM_autocorr_masks(maf, mat)
        
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
                        bl_time_mask = self.get_time_mask(baseline_data)
                        #print("Shape of time_mask = ", time_mask.shape)

                        if self.isMasked:
                            baseline_data[..., bl_time_mask].data = 0
                            baseline_data[..., bl_time_mask].mask = True
                        else:
                            baseline_data[..., bl_time_mask] = 0

                        crosscorr_masks[str(ibl) + 't'] = bl_time_mask

                    if mcf:
                        bl_freq_mask = self.get_freq_mask(baseline_data)
                        if self.isMasked:
                            baseline_data[bl_freq_mask, ...].data = 0
                            baseline_data[bl_freq_mask, ...].mask = True
                        else:
                            baseline_data[bl_freq_mask, ...] = 0

                        crosscorr_masks[str(ibl) + 'f'] = bl_freq_mask

                    block[ibl] = baseline_data
                    cas_sum += baseline_data


        if mcasf or mcast:
            #cas_sum = get_cas_sum_ma(block_dict)     #cas_sum should also be a masked array
            #Finally find bad samples in the CAS
            cas_masks['f'] = self.get_freq_mask(cas_sum)    #Currently the IQRM cannot support masked arrays, so passing only the data values
            if self.isMasked:
                cas_sum[cas_masks['f']].data = 0
                cas_sum[cas_masks['f']].mask = True
            else:
                cas_sum[cas_masks['f']] = 0


            cas_masks['t'] = self.get_time_mask(cas_sum)
            #We don't bother zero-ing the cas-sum now since it will not be needed any further
            #if self.isMasked:
                #cas_sum[..., cas_masks['t']].data = 0
                #cas_sum[..., cas_masks['t']].mask = True
            #else:
                #cas_sum[..., cas_masks['t']] = 0

            if self.isMasked:
                block[:, cas_masks['f'], ...].data = 0
                block[:, cas_masks['f'], ...].mask = True

                block[..., cas_masks['t']].data = 0
                block[..., cas_masks['t']].mask = True

            else:
                block[:, cas_masks['f'], ...] = 0
                block[..., cas_masks['t']] = 0

        return block, autocorr_masks, crosscorr_masks, cas_masks


