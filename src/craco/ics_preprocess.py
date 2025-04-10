import numpy as np
from numba import njit
import IPython
from iqrm import iqrm_mask

def get_ics_masks(ics_data, new_tf_weights, subnt=128, nsubf = 8, zap_along_freq = False, freq_radius = None, freq_threshold = 5, zap_along_time = True, time_radius = None, time_threshold=5):
    '''
    This function takes in a block of ICS data - derives new tf mask from it after taking into account
    the existing fixed_freq_weights and input_tf_weights, and ORs the newly derived weights with the input_tf_weights
    Returns output_tf_weights

    ics_data: np.ndarray
            Block of ICS data - simple numpy array of dtyple float

    new_tf_weights: np.ndarray, dtype=bool
            Array in which new_tf_weights will be saved
    subnt: int
            Number of samples in a sub-block that will be used to zap along time axis
    nsubf: int 
            Number of sub-bands to divide the block into to zap along freq axis
    zap_along_freq: bool
            Zap along frequency axis True/False
    freq_radius: int
            Number of freq bins to consider when finding outliers in freq
    freq_threshold: float
            Threshold to use to find outliers in freq
    zap_along_time: bool
            Zap aling the time axis True/False
    time_radius: int
            Number of time bins to consider when finding outliers in time
    time_threshold: float
            Threshold to use to find outliers in time
    '''

    #assert input_tf_weights.dtype == 'bool'
    #assert fixed_freq_weights.dtype == 'bool'
    try:
        nf, nt = ics_data.shape
        assert nt%subnt == 0
        assert nf%nsubf == 0
        #assert len(fixed_freq_weights) == nf
        #assert input_tf_weights.shape == ics_data.shape

        #ics_data[~fixed_freq_weights, :] = 0
        #ics_data[~input_tf_weights] = 0

        #new_tf_weights = np.ones_like(ics_data, dtype='bool')
        new_tf_weights[:] = True    #Reset the weights before computing new ones
        for ii in range(nt//subnt):
            tslice = slice(subnt*ii, subnt*(ii+1))
            if zap_along_freq:
                f_rms = ics_data[:, tslice].std(axis=1)
                if freq_radius is None:
                    freq_radius = nf // 4
                f_mask = iqrm_mask(f_rms, freq_radius, freq_threshold)[0]
                new_tf_weights[f_mask, tslice] = False

            if zap_along_time:
                if zap_along_freq:
                    #If we have zapped along freq as well, let's apply the newly derived weights first
                    ics_data[~new_tf_weights] = 0
                
                subnf = nf // nsubf
                if time_radius is None:
                    time_radius = subnt // 2

                for isubf in range(nsubf):
                    fslice = slice(isubf * subnf, (isubf+1)*subnf, 1)
                    d_subband = ics_data[fslice, tslice]
                    subband_mean = d_subband.mean(axis=0)
                
                    t_mask = iqrm_mask(subband_mean, time_radius, time_threshold)[0]
                    new_tf_weights[fslice, tslice][:, t_mask] = False
                    ics_data[fslice, tslice][:, t_mask] = 0

                t_total = ics_data[:, tslice].mean(axis=0)
                t_mask = iqrm_mask(t_total, time_radius, time_threshold)[0]
                new_tf_weights[:, tslice][:, t_mask] = False
    except Exception as e:
        #print(e)
        raise
        #IPython.embed()

    #return new_tf_weights #| input_tf_weights

@njit(parallel = True, cache = True)
def downsample_masks(tf_weights, downsampled_tf_weights):
    nf, nt = tf_weights.shape
    desired_nf, desired_nt = downsampled_tf_weights.shape

    ff = nf // desired_nf
    ft = nt // desired_nt
    
    for ii in prange(desired_nf):
        for jj in range(desired_nt):
            downsampled_tf_weights[ii, jj] = np.all(tf_weights[ii * ff : (ii+1) * ff, jj * ft : (jj+1)*ft])

