import numpy as np
from craco import uvfits_meta
from craft import fdmt
import argparse

def main():
    f = uvfits_meta.open(args.uvname)
    fch1 = f.freq_config.fch1
    foff = f.freq_config.foff
    nchan = f.freq_config.nchan

    fbottom = fch1 - foff
    thefdmt = fdmt.Fdmt(f_min = fbottom, f_off = foff, n_f = nchan, max_dt = args.dm_samps + 1, n_t = 256)

    true_eff_sigma = np.sqrt(thefdmt.get_eff_var_recursive(args.dm_samps, args.boxcar_w))
    applied_eff_sigma = np.sqrt(nchan)

    correction_multiplier = applied_eff_sigma / true_eff_sigma

    true_snr = args.det_snr * correction_multiplier

    print(args.det_snr, correction_multiplier, true_snr)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("-snr", "--det_snr", type=float, required=True, help="Original detection S/N")
    a.add_argument("-dm", "--dm_samps", type=int, required=True, help="Original detection DM in sample units")
    a.add_argument("-w", "--boxcar_w", type=int, required=True, help="Original detection boxcar width")
    a.add_argument("-uv", "--uvname", type=str, required=True, help="Path to the uvfits file containing the FRB")
    args = a.parse_args()
    main()
