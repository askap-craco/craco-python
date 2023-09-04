from craft import uvfits_snippet, craco
from craco.preprocess import Dedisp
import argparse


def main(args):
    print("Instantiating the UvfitsSnippet")
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, args.tstart, args.tend)
    print("Instantiating the Dedisp class")
    if args.dm_samps is None:
        ddp = Dedisp(f.uvsource.channel_frequencies, f.uvsource.tsamp.value, dm_pccc = args.dm_pccc)
    else:
        ddp = Dedisp(f.uvsource.channel_frequencies, f.uvsource.tsamp.value, dm_samps = args.dm_samps)

    print("Reading data from the uvfits file")
    data, _ = f.read_as_data_block_with_uvws()
    print("Running the dedisperse function")
    ddpout = ddp.dedisperse(0, craco.bl2array(data))
    print("Swapping the original data with de-dispersed data")
    f.swap_with_data(ddpout)
    print("Saving the data as a uvfits file")
    f.save(args.uvpath.replace(".uvfits", "ddp_dm{dm}.t{ss}_{es}.uvfits".format(dm=args.dm_samps, ss = f.start_samp, es = f.end_samp)), True)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to read from")
    a.add_argument("-tstart", type=int, help="Start sample (inclusive)", required = True)
    a.add_argument("-tend", type=int, help="End sample (inclusive)", required = True)
    g = a.add_mutually_exclusive_group(required = True)
    g.add_argument("-dm_samps", type=int, help="DM in sample units to de-disperse to", default=None)
    g.add_argument("-dm_pccc", type=float, help="DM in pc/cc to de-disperse to", default=None)
    
    args = a.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)


