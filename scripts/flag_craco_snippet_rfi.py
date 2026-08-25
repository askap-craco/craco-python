#!/bin/usr/env python

from casacore import tables
import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CracoSnippetMeasurementSet:
    def __init__(self, mspath):
        self.mspath = mspath
        self.mstab = tables.table(mspath, readonly=True)
        self._parse_basic_info()
        self._get_data()
        self._get_freqs()

    def _parse_basic_info(self, nr=1000):
        self._parse_antenna_info(nr=nr)
        self._parse_time_info()

    def _parse_antenna_info(self, nr=1000):
        ant1 = self.mstab.getcol("ANTENNA1", nrow=nr)
        ant2 = self.mstab.getcol("ANTENNA2", nrow=nr)
        antpair = set(list(zip(ant1, ant2)))
        nant1 = len(set(ant1))
        nant2 = len(set(ant2))
        assert nant1 == nant2, "Number of antennas in ANTENNA1 and ANTENNA2 columns do not match"
        self.nant = nant1
        self.nbl = len(antpair)

    def _parse_time_info(self):
        times = np.unique(self.mstab.getcol("TIME"))
        self.nt = len(times)

    def _get_data(self):
        data = self.mstab.getcol("DATA")
        self.nchan = data.shape[1]
        self.npol = data.shape[2]
        rdata = data.reshape((self.nt, self.nbl, self.nchan, self.npol))
        self.data = rdata

    def _get_freqs(self):
        self.spwtab = tables.table(f"{self.mspath}::SPECTRAL_WINDOW", readonly=True)
        self.freqs = self.spwtab.getcol("CHAN_FREQ") / 1e6

def create_mask(xx, yy, niter=5, nsigma=3):

    mask = np.ones(len(yy), dtype=bool)

    for _ in range(niter):
        # Fit line to current inliers
        p = np.polyfit(xx[mask], yy[mask], 1)

        # Residuals
        residuals = yy - np.polyval(p, xx)

        # Estimate scatter robustly using MAD
        median = np.median(residuals[mask])
        mad = np.median(np.abs(residuals[mask] - median))
        sigma = 1.4826 * mad

        # Reject points > 5 sigma from the line
        mask = np.abs(residuals - median) < nsigma * sigma
    return mask

def expand_mask(mask, flag=True, window=3):
    """
    Expand a boolean mask by a given window size.

    Parameters:
    - mask: 1D boolean array to be expanded.
    - flag: If True, expand the True values; if False, expand the False values.
    - window: Number of elements to expand on each side.

    Returns:
    - expanded_mask: 1D boolean array with the same shape as mask, expanded.
    """
    expanded_mask = np.copy(mask)
    for i in range(len(mask)):
        if mask[i] == flag:
            start = max(0, i - window)
            end = min(len(mask), i + window + 1)
            expanded_mask[start:end] = flag
    return expanded_mask

def flag_freqs_string(mask, freqs):
    """
    Generate a string representation of flagged frequency ranges.

    Parameters:
    - mask: 1D boolean array where True indicates a flagged frequency.
    - freqs: 1D array of frequencies corresponding to the mask.

    Returns:
    - flag_string: String representation of flagged frequency ranges.
    """
    flag_ranges = []
    in_flag = False
    start_freq = None

    for i in range(len(mask)):
        if mask[i] and not in_flag:
            # Start of a new flagged range
            in_flag = True
            start_freq = freqs[i]
        elif not mask[i] and in_flag:
            # End of the current flagged range
            in_flag = False
            end_freq = freqs[i - 1]
            flag_ranges.append(f"{start_freq-1:.0f}~{end_freq+1:.0f}MHz")

    # Handle case where the last frequency is flagged
    if in_flag:
        end_freq = freqs[-1]
        flag_ranges.append(f"{start_freq-1:.0f}~{end_freq+1:.0f}MHz")

    return ','.join(flag_ranges)

def get_flag_freqs(ms):
    """
    Get the flagged frequency ranges from a CracoSnippetMeasurementSet.

    Parameters:
    - ms: An instance of CracoSnippetMeasurementSet.

    Returns:
    - flag_string: String representation of flagged frequency ranges.
    """
    if isinstance(ms, str):
        ms = CracoSnippetMeasurementSet(ms)
    data = ms.data
    waterfall = np.nanmean(np.abs(data[..., 0]), axis=1)
    waterfall[waterfall == 0] = np.nan
    freqsed = np.nanmean(waterfall, axis=0)

    # Create mask for flagged frequencies
    mask = ~create_mask(ms.freqs[0], freqsed)
    
    # Expand the mask to include neighboring frequencies
    expanded_mask = expand_mask(mask, flag=True, window=3)

    # Generate string representation of flagged frequency ranges
    flag_string = flag_freqs_string(expanded_mask, ms.freqs[0])
    
    return flag_string

def run(fieldms, burstms):
    """
    Run the RFI diagnostic on the provided measurement sets, and do flagging"""
    from casatasks import flagdata

    flagstr = get_flag_freqs(fieldms)
    logger.info(f"Flagging frequencies: {flagstr}")

    flagdata(vis=fieldms, spw=f"*:{flagstr}", mode='manual', action='apply')
    flagdata(vis=burstms, spw=f"*:{flagstr}", mode='manual', action='apply')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flag RFI in CRACO measurement sets.")
    parser.add_argument("--fieldms", type=str, help="Path to the field measurement set.")
    parser.add_argument("--burstms", type=str, help="Path to the burst measurement set.")

    args = parser.parse_args()

    run(args.fieldms, args.burstms)