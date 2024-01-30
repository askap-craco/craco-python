from craco import uvfits_meta, preprocess
from craft import craco_plan
from craco.uvfits_writer import UvfitsWriter, copy_visparams_to_visrow
import numpy as np
import argparse

def main(args):
    assert args.tend >= args.tstart
    nsamps_to_read = args.tend - args.tstart + 1
    f = uvfits_meta.open(args.uvpath, metadata_file = args.metadata, skip_blocks = args.tstart, mask=args.apply_metadata_masks)

    if args.outname:
        outname = args.outname
    else:
        outname = args.uvpath.split("/")[-1].strip(".uvfits") + f"_ex_{args.tstart}_{args.tend}.uvfits"
    of = UvfitsWriter(outname, infile=args.uvpath)
    of.copy_header()

    if args.dedisp_pccc:
        ddp = preprocess.Dedisp(f.channel_frequencies, 
                                f.tsamp.value,
                                dm_pccc = args.dedisp_pccc)


    if args.calib:
        print(f"Starting calibration using {args.calib}")
        plan_args = " "
        if args.flag_ants:
            plan_args += f"--flag-ants {args.flag_ants}"

        plan = craco_plan.PipelinePlan(f, plan_args)
        calibrator = preprocess.Calibrate(plan = plan,
                                block_dtype=np.ndarray,
                                miriad_gains_file=args.calib,
                                baseline_order=f.internal_baseline_order)
    
    for iblk, visout in enumerate(f.fast_raw_blocks(nsamp = nsamps_to_read, nt = 1, raw_date=True)):
        data_block = f.convert_visrows_into_block(visout)
        print(f"Shape of data_block is {data_block.shape}")
        #modified_data = data_block.copy()
        if args.calib:
            data_block = calibrator.apply_calibration(data_block[:, 0, 0, 0, :, 0, :])
        if args.sky_subtract:
            data_block = preprocess.normalise(data_block)
        if args.dedisp_pccc:
            data_block = preprocess.fill_masked_values(data_block, fill_value = 0)
            data_block = ddp.dedisperse(iblk, inblock=data_block)

        #print(f"Shape of data_block after processing is {data_block.shape}, type is {type(data_block)}, mask is {data_block.mask}")

        modified_visdata = f.convert_block_into_visrows(data_block)
        UU = visout['UU'].flatten()
        VV = visout['VV'].flatten()
        WW = visout['WW'].flatten()
        DATE = visout['DATE'].flatten()
        BASELINE = visout['BASELINE'].flatten()
        FREQSEL = visout['FREQSEL'].flatten()
        SOURCE = visout['SOURCE'].flatten()
        INTTIM = visout['INTTIM'].flatten()
        visout = copy_visparams_to_visrow(modified_visdata, UU, VV, WW, DATE, BASELINE, FREQSEL, SOURCE, INTTIM)
        of.write_visrows_to_disk(visout)

    of.update_header()
    of.write_header()
    of.close_file(fix_length = True)
    of.append_supplementary_tables(uvsource = f)
    f.close()



if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to extract from")
    a.add_argument("-metadata", type=str, help="Path to the metadata file")
    a.add_argument("-apply-metadata-masks", type=bool, help="Apply metadata masks? (def=True)", default=True)
    a.add_argument("-tstart", type=int, help="Tstart in samples (def:0)", default=0)
    a.add_argument("-tend", type=int, help="Tend in samples (inclusive) (def:1)", default = 1)
    a.add_argument("-dedisp_pccc", type=int, help="Dedisperse by x DM units (pc/cc)", default = None)
    a.add_argument("-calib", type=str, help="Path to the calibration soln", default=None)
    a.add_argument("-sky_subtract", action='store_true', help="Run sky subtraction on the data (def:False)", default=False)
    a.add_argument("--flag-ants", type=str, help="Flag these ants", default=None)
    a.add_argument("-outname", type=str, help="Name of the output file", default=None)

    args = a.parse_args()
    main(args)