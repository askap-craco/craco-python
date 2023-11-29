from craco import uvfits_snippet
from craft.craco import bl2array
import argparse
import numpy as np

def convert_uvws_list_to_array(uvws_list):
    uvws = {}
    dtype='>f8'
    nt = len(uvws_list)
    for blid in list(uvws_list[0].keys()):
        uvws[blid] = np.zeros((3, nt), dtype='f8')

        #          ('UU', uvw_list[blid]['UU'].dtype),
        #          ('VV', uvw_list[blid]['UU'].dtype),
        #          ('WW', uvw_list[blid]['WW'].dtype)
        #       ])


    for it in range(nt):
        for blid in list(uvws_list[0].keys()):

            uvws[blid][0, it] = uvws_list[it][blid]['UU']
            uvws[blid][1, it] = uvws_list[it][blid]['VV']
            uvws[blid][2, it] = uvws_list[it][blid]['WW']

    return uvws



def main():
    args = get_parser()
    f = uvfits_snippet.UvfitsSnippet(args.uvpath, start_samp=args.start_samp, end_samp=args.end_samp, metadata_file=args.metadata_file)
    if args.outname== None:
        outname = args.uvpath.replace(".uvfits", ".attached.uvfits")
    else:
        outname = args.outname

    if args.use_time_blocks:    
        print("Using time_block_with_uvws()")
        data, uvws = f.read_as_data_block_with_uvws()
        f.swap_with_data(bl2array(data), bl2array(uvws, dtype=np.float64))

    elif args.use_fast_time_blocks:
        print("Using fast_time_blocks()")
        fast_blocker = f.uvsource.fast_time_blocks(nt = args.end_samp - args.start_samp + 1, fetch_uvws=True)
        data, uvws = next(fast_blocker)
        uvws = convert_uvws_list_to_array(uvws)
        f.swap_with_data(data[:, 0, 0, 0], bl2array(uvws))
    elif args.use_visrows:
        print("Using visrows()")
        pass

    f.save(outname=outname, overwrite=True)

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the raw uvfits file")
    a.add_argument("metadata_file", type=str, help="Path to the metadata file from which to fetch uvw values")
    a.add_argument("-outname", type=str, help="Name of the output filename (def=*.uvw.uvfits)", default=None)
    a.add_argument("-start_samp", type=int, help="Starting sample number (Def=0)", default=0)
    a.add_argument("-end_samp", type=int, help="Last sample number (inclusive range); say -1 to go to the end of the file (Def=-1)", default=-1)
    g = a.add_mutually_exclusive_group()
    g.add_argument("-use_time_blocks", action='store_true', help="Use the time_block_with_uvw_range() instead of faster uvsource.vis() (def:False)", default=False)
    g.add_argument("-use_fast_time_blocks", action='store_true', help="Use the fast_time_block() (def:True)", default=True)
    g.add_argument("-use_visrows", action='store_true', help="Use visrows to read the data (def:False)", default=False)


    args = a.parse_args()
    return args


if __name__ == '__main__':
    main()
