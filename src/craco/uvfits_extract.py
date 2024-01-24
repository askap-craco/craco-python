from craco import uvfits_meta
from craco.uvfits_writer import UvfitsWriter
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

    for iblk, visout in enumerate(f.fast_raw_blocks(nsamp = nsamps_to_read, nt = 1, raw_date=True)):
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
    a.add_argument("-outname", type=str, help="Name of the output file", default=None)

    args = a.parse_args()
    main(args)