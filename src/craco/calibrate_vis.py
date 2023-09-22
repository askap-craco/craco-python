from craft import uvfits_snippet, craco, craco_plan
from craco.preprocess import Calibrate
import argparse
import numpy as np

def main(args):
    print("Instantiating the UvfitsSnippet")
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, args.tstart, args.tend)
    print("Instantiating the Calibration class")
    plan = craco_plan.PipelinePlan(f.uvsource, " ")
    calibrator = Calibrate(plan = plan, block_dtype=np.ndarray, 
            miriad_gains_file = args.cal,
            baseline_order = plan.baseline_order)
    print("Reading data from the uvfits file")
    data, _ = f.read_as_data_block_with_uvws()
    print("Running the calibrate function")
    caldata = calibrator.apply_calibration(craco.bl2array(data))
    print("Swapping the original data with calibrated data")
    #if caldata.dtype != np.complex64 and args.force_dtype:
    #    caldata = caldata.astype(np.complex64, casting='unsafe')
    f.swap_with_data(caldata)
    print("Saving the data as a uvfits file")
    if args.outname is not None:
        outname = args.outname
    else:
        basename = args.uvpath.split("/")[-1]
        outname = basename.replace(".uvfits", ".cal.uvfits")

    f.save(outname, True)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to read from")
    a.add_argument("-tstart", type=int, help="Start sample (inclusive), (def:0)", default=0)
    a.add_argument("-tend", type=int, help="End sample (inclusive), say -1 to indicate full file (def:-1)", default=-1)
    a.add_argument("-outname", type=int, help="Name of the output vis", default = None)
    a.add_argument("-cal", type=str, help="Path to the calibration soln", default=None, required = True)
    a.add_argument("-force_dtype", action='store_true', help="Try doing an unsafe casting of data to match the data type from the file (def:False)", default=False)
    args = a.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)


