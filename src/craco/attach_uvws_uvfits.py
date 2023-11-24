from craco import uvfits_snippet
from craft.craco import bl2array
import argparse
import numpy as np

def main():
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, start_samp=args.start_samp, end_samp=args.end_samp, metadata_file=args.metadata_file)
    if args.outname== None:
        outname = args.uvpath.replace(".uvfits", ".uvw.uvfits")
    else:
        outname = args.outname

    data, uvws = f.read_as_data_block_with_uvws()
    f.swap_with_data(bl2array(data), bl2array(uvws, dtype=np.float))
    f.save(outname=outname, overwrite=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the raw uvfits file")
    a.add_argument("metadata_file", type=str, help="Path to the metadata file from which to fetch uvw values")
    a.add_argument("-outname", type=str, help="Name of the output filename (def=*.uvw.uvfits)", default=None)
    a.add_argument("-start_samp", type=int, help="Starting sample number (Def=0)", default=0)
    a.add_argument("-end_samp", type=int, help="Last sample number (inclusive range); say -1 to go to the end of the file (Def=-1)", default=-1)

    args = a.parse_args()
    main()
