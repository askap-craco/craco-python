from craft import craco
from craco import uvfits_snippet
from craco.preprocess import Dedisp
import argparse


def main(args):
    print("Instantiating the UvfitsSnippet")
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, args.tstart, args.tend, args.metadata_file)
    print("Instantiating the Dedisp class")
    if args.dm_samps is None:
        ddp = Dedisp(f.uvsource.channel_frequencies, f.uvsource.tsamp.value, dm_pccc = args.dm_pccc)
        outname_dm_part = f"dm_pccc{args.dm_pccc}"
    else:
        ddp = Dedisp(f.uvsource.channel_frequencies, f.uvsource.tsamp.value, dm_samps = args.dm_samps)
        outname_dm_part = f"dm_samps{args.dm_samps}"

    print("Reading data from the uvfits file")
    data, _ = f.read_as_data_block_with_uvws()
    print("Running the dedisperse function")
    ddpout = ddp.dedisperse(0, craco.bl2array(data))
    print("Swapping the original data with de-dispersed data")
    f.swap_with_data(ddpout)
    print("Saving the data as a uvfits file")
    if args.outname is not None:
        outname = args.outname
    else:
        basename = args.uvpath.split("/")[-1]
        outname = basename.replace(".uvfits", ".{dm}.t{ss}_{es}.uvfits".format(dm=outname_dm_part, ss = f.start_samp, es = f.end_samp))

    f.save(outname, True)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to read from")
    a.add_argument("-tstart", type=int, help="Start sample (inclusive), def:0", default=0)
    a.add_argument("-tend", type=int, help="End sample (inclusive), say -1 for end of file, (def:-1)", default=-1)
    a.add_argument("-outname", type=int, help="Name of the output vis", default = None)
    g = a.add_mutually_exclusive_group(required = True)
    g.add_argument("-dm_samps", type=int, help="DM in sample units to de-disperse to", default=None)
    g.add_argument("-dm_pccc", type=float, help="DM in pc/cc to de-disperse to", default=None)
    a.add_argument('-metadata-file', type=str, help='Metadata file to use for UVws instead of defaults', default=None)
    
    args = a.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)


