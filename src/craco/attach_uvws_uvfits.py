from craco import uvfits_snippet
import argparse


def main():
    f = uvfits.UvfitsSnippet(args.uvpath, start_samp=0, end_samp=-1, metadata_file=args.metadata_file)
    if args.outname = None:
        outname = args.uvpath.replace(".uvfits", ".uvw.uvfits")
    else:
        outname = args.outname

    data, uvws, (ss, se) = f.read_as_data_block_with_uvws()
    f.swap_with_data(data, uvws)
    f.save(outname=outname, overwrite=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the raw uvfits file")
    a.add_argument("metadata_file", type=str, help="Path to the metadata file from which to fetch uvw values")
    a.add_argument("-outname", type=str, help="Name of the output filename (def=*.uvw.uvfits)", default=None)

    args = a.parse_args()
    main()
