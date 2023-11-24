from craft import craco
from craco import uvfits_snippet
from craco.preprocess import normalise
import argparse


def main(args):
    print("Instantiating the UvfitsSnippet")
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, args.tstart, args.tend, args.metadata_file)

    print("Reading data from the uvfits file")
    data, _ = f.read_as_data_block_with_uvws()
    print("Running the normalise function")
    normdata = normalise(craco.bl2array(data))
    print("Swapping the original data with normalised data")
    f.swap_with_data(normdata)
    print("Saving the data as a uvfits file")
    if args.outname is not None:
        outname = args.outname
    else:
        basename = args.uvpath.split("/")[-1]
        outname = basename.replace(".uvfits", "norm.uvfits")

    f.save(outname, True)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to read from")
    a.add_argument("-tstart", type=int, help="Start sample (inclusive), (def:0)", default=0)
    a.add_argument("-tend", type=int, help="End sample (inclusive), say -1 for end of file (def:-1)", default=-1)
    a.add_argument("-outname", type=str, help="Name of the output vis", default = None)
    a.add_argument('-metadata-file', type=str, help='Metadata file to use for UVws instead of defaults', default=None)
    args = a.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)


