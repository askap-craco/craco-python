from craco import uvfits_meta
from craco.uvfits_writer import UvfitsWriter, copy_visparams_to_visrow
import argparse

def main(args):
    assert args.tend >= args.tstart
    nsamps_to_read = args.tend - args.tstart + 1
    f = uvfits_meta.open(args.uvpath, metadata_file = args.metadata, skip_blocks = args.tstart, mask=args.apply_metadata_masks)

    if args.outname:
        outname = args.outname
    else:
        outname = args.uvpath.split("/")[-1].strip(".uvfits") + f"_ex_{args.tstart}_{args.tend}_tx_{args.tx}.uvfits"
    of = UvfitsWriter(outname, infile=args.uvpath)
    of.copy_header()

    avg_vis = None
    avg_block = None
    navg = 0

    elements_to_modify = ['UU', 'VV', 'WW', 'DATE']
    for iblk, visout in enumerate(f.fast_raw_blocks(nsamp = nsamps_to_read, nt = 1, raw_date=True)):
        #Convert the visrows into blocks
        this_block = f.convert_visrows_into_block(visout, keep_all_baselines = True)

        if avg_vis is None:
            #Load the current vis and block as Averaged vis and blocks
            avg_vis = visout
            navg = 1

            if args.mask_conservatively:
                #mask_conservatively implies that we want the sum to be masked if any of the elements are masked. 
                #Therefore, we keep the avg_block as a masked array, and when we add a masked element, the whole output gets masked
                avg_block = this_block
            else:
                #Otherwise, we convert to a regular array, but we have to keep track of how many unmasked elements have we added
                avg_block = this_block.filled(fill_value = 0)
                #Setup the coutner for the number of valid elements averaged in data
                nsum = np.zeros(avg_block.shape, dtype='uint32') + ~this_block.mask
        else:
            #Add all the elements that need to be modified
            for element in elements_to_modify:
                avg_vis[element] += visout[element]
            
            navg += 1

            #Add up the data elements
            if args.mask_conservatively:
                avg_block += this_block
            else:
                avg_block += this_block.filled(fill_value = 0)
                nsum += ~this_block.mask


        #If we have added up args.tx samples - 
        if navg == args.tx:
            #Divide the added elements by navg
            for element in elements_to_modify:
                avg_vis[element] /= navg

            #Update the Integration time
            avg_vis['INTTIM'] = navg * visout['INTTIM']

            #Divide the data by the number of valid elements added
            if args.mask_conservatively:
                #For sections of data averaged which had even a single element masked, the output would be masked too,
                #So it doesn't really matter what you divide by here. This only matters for sections which had all 
                #good elements in it, so you would want to divide by the full length of the section, which is given by navg.
                avg_block /= navg
            else:
                avg_block /= nsum
            
            #Convert the block back into visrows
            avg_visout = f.convert_block_to_visrows(avg_block)
            #Add the visparams to the converted visrows
            avg_visout = copy_visparams_to_visrow(avg_visout, avg_vis['UU'], avg_vis['VV'], avg_vis['WW'], avg_vis['DATE'], avg_vis['BASELINE'], avg_vis['FREQSEL'], avg_vis['SOURCE'], avg_vis['INTTIM'])

            #Write them out
            of.write_visrows_to_disk(avg_visout)

            #Reset the avg variables
            avg_vis = None
            avg_block = None
        
        if iblk >= args.tend - args.tstart:
            break
            #This break means that the last set of samples which did not add up tx times will be left out from the output file

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
    a.add_argument("-mask_conservatively", action='store_true', help="Mask the output if any of the inputs are masked? (def=False)", default=False)
    a.add_argument("-tstart", type=int, help="Tstart in samples (def:0)", default=0)
    a.add_argument("-tend", type=int, help="Tend in samples (inclusive) (def:1)", default = 1)
    a.add_argument("-tx", type=int, help="Averaging factor (int)", default= None, required = True)
    a.add_argument("-outname", type=str, help="Name of the output file", default=None)

    args = a.parse_args()
    main(args)
