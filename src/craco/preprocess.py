from . import calibration
from iqrm import iqrm_mask
import numpy as np
from craft.craco import bl2ant

def normalise(block, target_input_rms = 1):
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

    target_input_rms: float
            Desired std of each channel
    '''
    
    if type(block) == dict:
        new_block = {}
        for ibl, bldata in block.items():
            print(f"====>> The shape of received block[ibl] for ibl{ibl} is {block[ibl].shape}")
            existing_rms = bldata.std(axis=-1) / np.sqrt(2)
            new_block[ibl] = (bldata - bldata.mean(axis=-1, keepdims=True)) * (target_input_rms / existing_rms)[..., None]
            print(f"====>> The shape of normalised block[ibl] for ibl{ibl} is {block[ibl].shape}")
    elif type(block) == np.ndarray or type(block) == np.ma.core.MaskedArray:
        existing_rms = block.std(axis=-1) / np.sqrt(2)   
        new_block = (block - block.mean(axis=-1, keepdims = True)) * (target_input_rms / existing_rms)[..., None]
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



class Calibrate:
    def __init__(self, block_dtype, miriad_gains_file, baseline_order, keep_masks = True):
        self.gains_file = miriad_gains_file
        self.baseline_order = baseline_order
        self.reload_gains()

        if block_dtype not in [np.ndarray, np.ma.core.MaskedArray, dict]:
            raise ValueError("Unknown dtype of block provided")

        self.block_dtype = block_dtype
        self.keep_masks = keep_masks
        

    def reload_gains(self):
        self.ant_gains = calibration.load_gains(self.gains_file)
        self.gains_array = calibration.soln2array(self.ant_gains, self.baseline_order)
        self.gains_pol_avged_array = self.gains_array.mean(axis=-2, keepdims=True)
        if type(self.gains_array) == np.ma.core.MaskedArray:
            self.sol_isMasked = True
        else:
            self.sol_isMasked = False

    def apply_calibration(self, block):
        block_isMasked, nPol, block_type = get_isMasked_nPol(block)
        assert block_type == self.block_dtype, f"You Liar!, {block_type}, {self.block_dtype}"

        if block_type == dict:
            print(f"Baseline order is: {self.baseline_order}")
            for ibl, blid in enumerate(self.baseline_order):
                print(f"ibl {ibl}, blid={blid}, npol= {nPol}, bldata.shape = {block[blid].shape}, gains_array.shape = {self.gains_array.shape, self.gains_pol_avged_array.shape}")
                if nPol == 2:
                    block[blid] = self.gains_array[ibl, ...] * block[blid] 
                elif nPol ==1:
                    block[blid] =  self.gains_pol_avged_array[ibl, ...] * block[blid]
                elif nPol ==0:
                    block[blid] = self.gains_pol_avged_array[ibl, ...].squeeze() * block[blid]
                else:
                    raise ValueError(f"Expected nPol 0, 1, or 2, but got {nPol}")
                
                print(f"~~~~~~~~~~ 'apply_calibration() says' The shape of block[{blid}] is {block[blid].shape}")
                
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
                calibrated_block = self.gains_pol_avged_array.squeeze() * block
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
            print("Shape of baseline_data given to me is=  ",baseline_data.shape)
            rms = baseline_data.mean(axis=1).std(axis=0).squeeze()
        else:
            print("Shape of baseline_data given to me is=  ~~~~~~",baseline_data.shape)
            rms = baseline_data.std(axis=0).squeeze()

        print("LEN(RMS)  =  ", len(rms))
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
        print("Cleaning autocorr for ibl", ibl, "the ant is going to be ", bl2ant(self.baseline_order[ibl]))
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

            print("autocorr masks are", autocorr_masks)
        
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
                        print("Shape of baseline_data = ", baseline_data.shape)
                        bl_time_mask = self.get_time_mask(baseline_data)
                        print("Shape of time_mask = ", bl_time_mask.shape)

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
