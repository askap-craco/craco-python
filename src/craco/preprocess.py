from . import rfi_removal, calibration
import numpy as np

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

class Calibrate:
    def __init__(self, miriad_gains_file, block_dtype):
        self.gains_file = miriad_gains_file
        self.reload_gains()
        if self.gains.ndim == 3:
            self.pol_axis_exists = True
        elif self.gains.ndim == 2:
            self.pol_axis_exists = False
        else:
            raise Exception(f"Gains array has unexecpted ndim {self.gains.ndim}")

        self.block_dtype = block_dtype



    def reload_gains(self):
        self.gains = calibration.load_gains(self.gains_file)

    def apply_calibration(block):
    
        return

def remove_RFI(block, mask_autos_freq = True, mask_corrs_freq = True, mask_cas_freq = True, mask_autos_time = False, mask_corrs_time = False, mask_cas_time = True):

    rfi_removal.run_IQRM_cleaning(block, 
    maf = mask_autos_freq,
    mat = mask_autos_time,
    mcf = mask_corrs_freq,
    mct = mask_corrs_time,
    mcasf = mask_cas_freq,
    mcast = mask_cas_time)

