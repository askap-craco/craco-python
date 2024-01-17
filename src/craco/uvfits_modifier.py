from craft import craco, craco_plan
from craft.cmdline import strrange
from craco import preprocess, uvfits_snippet
import argparse
import numpy as np

def main(args):
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, args.tstart, args.tend, metadata_file = args.metadata)
    outname = args.uvpath.strip(".uvfits")

    if args.metadata:
        outname += ".uvw"

    swap_later = False
    if args.calib or args.sky_subtract or args.dedisp:
        print(f"Reading the data from samp {args.tstart} to {args.tend}")
        data, uvws = f.read_as_data_block_with_uvws()
        data = craco.bl2array(data)
        swap_later = True
        print(f"Done reading")
    
    if args.calib:
        print(f"Starting calibration using {args.calib}")
        plan_args = " "
        if args.flag_ants:
            plan_args += f"--flag-ants {args.flag_ants}"

        plan = craco_plan.PipelinePlan(f.uvsource, plan_args)
        calibrator = preprocess.Calibrate(plan = plan,
                                block_dtype=np.ndarray,
                                miriad_gains_file=args.calib,
                                baseline_order=f.uvsource.raw_baseline_order)

        data = calibrator.apply_calibration(data)
        outname += ".calib"
        print(f"Done calibrating")

    if args.sky_subtract:
        print(f"Starting sky subtraction using preprocess.normalise")
        data = preprocess.normalise(data)
        outname += ".skysub"
        print(f"Finished sky subtraction")

    if args.dedisp:
        print(f"Starting de-dispersion at a DM of {args.dedisp} pc/cc")
        ddp = preprocess.Dedisp(f.uvsource.channel_frequencies, 
                                f.uvsource.tsamp.value,
                                dm_pccc = args.dedisp)
        outname += ".ddp_{args.dedisp:.1f}"
        print(f"Finished de-dispersion")
    
    if swap_later:
        f.swap_with_data(data)
    
    outname += ".uvfits"

    if args.outname:
        outname = args.outname

    f.save(outname, True)
    



def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to read from")
    a.add_argument('-metadata', type=str, help='Metadata file to use for UVws instead of defaults', default=None)
    a.add_argument("-tstart", type=int, help="Start sample (inclusive), (def:0)", default=0)
    a.add_argument("-tend", type=int, help="End sample (inclusive), say -1 to indicate full file (def:-1)", default=-1)
    a.add_argument("-outname", type=str, help="Name of the output vis (def:<uvpath>.<options>.uvfits)", default = None)
    a.add_argument("-calib", type=str, help="Path to the calibration soln", default=None)
    a.add_argument("-sky_subtract", action='store_true', help="Run sky subtraction on the data (def:False)", default=False)
    a.add_argument("-dedisp", type=float, help="DM value (pc/cc) to dedisperse the visibilities by", default=None)
    a.add_argument("--flag-ants", type=str, help="Flag these ants", default=None)
    
    args = a.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
