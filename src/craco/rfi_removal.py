
from iqrm import iqrm_mask
import numpy as np
from craft.craco import bl2ant

def get_isMasked_isPol(iterator):
    _, data = next(iterator)
    if type(data) == np.ndarray:
        isMasked = False
        ndim = data.ndim
    elif type(data) == np.ma.core.MaskedArray:
        isMasked = True
        ndim = data.ndim

    if ndim == 2:
        isPol = False
    elif ndim == 3:
        isPol = True
    else:
        raise Exception(f"ndim of a single baseline can only be 2 or 3, found {ndim}")
    return isMasked, isPol


class RFI_cleaner:
    def __init__(self, block_dtype, baseline_order = None):
        '''
        block_dtype: dtype (np.ndarry or np.ma.core.MaskedArray or dict)
                    The data type of the block that will be passed
        baseline_order: list or 1-D numpy array
                A list of the blids in the same order as they would
                be arranged in the block if it is an array. Not required 
                if block_dtype is a dict (where the block itself contains
                the blids as keys)
        '''
        if block_dtype not in [dict, np.ndarray, np.ma.core.MaskedArray]:
            raise ValueError("Only dicts, np.ndarrays and np.ma.core.MaskedArrays are currently supported")
   
        if block_dtype != dict and baseline_order is None:
            raise ValueError("I need blids for autocorr antid to bl mapping.")
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


    def get_IQRM_autocorr_masks(self, freq=True, time=True):
        autocorr_masks = {}
        for ibl, baseline_data in self.bl_iterator:
            a1, a2 = bl2ant(self.baseline_order[ibl])
            if a1 != a2:
                continue
            
            if freq:
                autocorr_freq_mask = self.get_freq_mask(np.abs(baseline_data))
                autocorr_masks[str(a1) + 'f'] = autocorr_freq_mask 
            if time:
                autocorr_time_mask = self.get_time_mask(np.abs(baseline_data))
                autocorr_masks[str(a1) + 't'] = autocorr_time_mask
        return autocorr_masks

    def clean_bl_using_autocorr_mask(self, ibl, bldata, autocorr_masks, freq, time):
        ant1, ant2 = bl2ant(self.baseline_order[ibl])
        if freq:
            autocorr_freq_mask = autocorr_masks[str(ant1) + 'f'] | autocorr_masks[str(ant2) + 'f']
            if self.isMasked:
                bldata[autocorr_freq_mask].mask = True
                bldata[autocorr_freq_mask].data = 0
            else:
                bldata[autocorr_freq_mask] = 0

        if time:
            autocorr_time_mask = autocorr_masks[str(ant1) + 't'] | autocorr_masks[str(ant2) + 't']
            if self.isMasked:
                bldata[..., autocorr_time_mask].mask = True
                bldata[..., autocorr_time_mask].data = 0
            else:
                bldata[..., autocorr_time_mask] = 0


    def run_IQRM_cleaning(self, block, maf, mat, mcf, mct, mcasf, mcast):
        '''
        Does the IQRM magic

        block needs to be a dictionary, indexed by blid, or a np.ndarray, or np.ma.core.MaskedArray
        Each baseline data should have shape (nf, npol, nt), even if npol == 1

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

        if type(block) == dict:
            self.bl_iterator = block.items()
            self.isMasked, self.pol_axis_exists = get_isMasked_isPol(self.bl_iterator)
            self.bl_iterator = block.items()    #Reset the iterator

            cas_sum = np.zeros_like(block[next(iter(block))])

        elif type(block) == np.ndarray:
            self.bl_iterator = enumerate(block)
            self.isMasked, self.pol_axis_exists = get_isMasked_isPol(self.bl_iterator)
            self.bl_iterator = enumerate(block)       #Reset the iterator

            cas_sum = np.zeros_like(block[0, ...])


        if maf or mat:
            autocorr_masks = self.get_IQRM_autocorr_masks(maf, mat)

        if maf or mat or mcf or mct or mcasf or mcast:
           for ibl, baseline_data in self.bl_iterator:

                if self.isMasked:
                    baseline_data.data[baseline_data.mask] = 0  #Zero the data where the data is already flagged, to avoid the IQRM getting too swayed by the bad samples

                if maf or mat:
                   self.clean_bl_using_autocorr_mask(ibl, baseline_data, autocorr_masks, freq=maf, time=mat)


                if mcf or mct or mcasf or mcast:

                    ant1, ant2 = bl2ant(self.baseline_order[ibl])
                    if ant1 == ant2:            
                       continue

                    if mct:
                        bl_time_mask = self.get_time_mask(np.abs(baseline_data))
                        #print("Shape of time_mask = ", time_mask.shape)

                        if self.isMasked:
                            baseline_data[..., bl_time_mask].data = 0
                            baseline_data[..., bl_time_mask].mask = True
                        else:
                            baseline_data[..., bl_time_mask] = 0

                        crosscorr_masks[str(ibl) + 't'] = bl_time_mask

                    if mcf:
                        bl_freq_mask = self.get_freq_mask(np.abs(baseline_data))
                        if self.isMasked:
                            baseline_data[bl_freq_mask, ...].data = 0
                            baseline_data[bl_freq_mask, ...].mask = True
                        else:
                            baseline_data[bl_freq_mask, ...] = 0

                        crosscorr_masks[str(ibl) + 'f'] = bl_freq_mask

                    if mcasf or mcast:
                        cas_sum += np.abs(baseline_data)

        if mcasf or mcast:
            #cas_sum = get_cas_sum_ma(block_dict)     #cas_sum should also be a masked array
            #Finally find bad samples in the CAS
            cas_masks['f'] = self.get_freq_mask(cas_sum.data)    #Currently the IQRM cannot support masked arrays, so passing only the data values
            if self.isMasked:
                cas_sum[cas_masks['f']].data = 0
                cas_sum[cas_masks['f']].mask = True
            else:
                cas_sum[cas_masks['f']] = 0


            cas_masks['t'] = self.get_time_mask(cas_sum.data)
            #We don't bother zero-ing the cas-sum now since it will not be needed any further
            #if self.isMasked:
                #cas_sum[..., cas_masks['t']].data = 0
                #cas_sum[..., cas_masks['t']].mask = True
            #else:
                #cas_sum[..., cas_masks['t']] = 0

            for ibl, baseline_data in self.bl_iterator:
                if self.isMasked:
                    baseline_data[cas_masks['f']].data = 0
                    baseline_data[cas_masks['f']].mask = True

                    baseline_data[..., cas_masks['t']] = 0
                    baseline_data[..., cas_masks['t']].mask = True

                else:
                    baseline_data[cas_masks['f']] = 0
                    baseline_data[..., cas_masks['t']] = 0

        return autocorr_masks, crosscorr_masks, cas_masks

