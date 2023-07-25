from astropy.io import fits
import numpy as np
import argparse, sys


def write_cands(cands, o):
   for icand in range(len(cands[0])):
       total_sample = cands[0][icand]
       mpix = cands[1][icand]
       lpix = cands[2][icand]
       snr = cands[3][icand]
       boxc_width = cands[4][icand]
       dm = cands[5][icand]

       o.write(f"{snr:.2f}\t{int(lpix):3d}\t{int(mpix):3d}\t{int(boxc_width):3d}\t{int(total_sample):8d}\t{int(dm):2d}")
       o.write("\n")


def get_mad(data):
    d = data.ravel()
    median = np.median(d)
    mad = np.median(np.abs(d - median))
    return mad, median


def find_peaks(data, threshold):
    mad_value, median = get_mad(data) 
    rms = mad_value * 1.4826
    print("Median, rms/std =  ", median, rms)
    d = data - median
    peak_locs = np.where(d >= (threshold * rms))
    peak_vals = d[peak_locs] / rms

    cands = list(peak_locs)
    cands.append(peak_vals)
    print("returning", cands)
    return cands

def main():
    f = fits.open(args.fname)
    data = f[0].data
    nsamples = data.shape[0]
    
    outname = args.fname.strip().split("/")[-1] + ".pycands"
    o = open(outname, 'w')
    o.write("# SNR\tlpix\tmpix\tboxc_width\ttotal_sample\tdm\n")
    o.close()
    o = open(outname, 'a')
    data_ibox = np.zeros_like(data)
    for ibox in range(args.nbox):
        data_ibox[ibox:] = data_ibox[ibox:] + data[:nsamples-ibox] 
        if ibox > 0:
            data_ibox[:ibox, ...] = 0
        ibox_cands = find_peaks(data_ibox, args.threshold)
        print("Found {0} cands in ibox {1}".format(len(ibox_cands[0]), ibox))
        if len(ibox_cands[0]) > 0:
            boxcar_values = np.ones(len(ibox_cands[0])) * ibox
            dm_values = np.zeros(len(ibox_cands[0]))
            ibox_cands.append(boxcar_values)
            ibox_cands.append(dm_values)
            print(len(ibox_cands), len(ibox_cands[0])) 
            write_cands(ibox_cands, o)
        print("Done writing cands")

    o.write("# This file was generated using {0}".format(" ".join(sys.argv)))
    o.close()



if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("fname", type=str, help="Path to the fits file name to process")
    a.add_argument("-threshold", type=float, help="Threshold to use (def = 6)", default=6)
    a.add_argument("-nbox", type=int, help="No of boxcar trials (def = 8)", default=8)

    args = a.parse_args()
    main()

