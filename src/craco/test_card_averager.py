#!/usr/bin/env python
from craco.card_averager import Averager
from craco.cardcap import CardcapFile, get_single_packet_dtype
from craco.cardcapmerger import CcapMerger
import numpy as np
import glob
import time

def test_averaging():
    cardfiles = glob.glob('/data/craco/ban115/craco-python/notebooks/data/SB43128/run3/1934_b07_c01+f?.fits')
    #assert len(cardfiles) == 6

    nt = 64
    nbeam = 36
    nbl = 465
    nant = 30
    nfpga = 6
    nc = 4*nfpga
    npol = 2
    polsum = npol == 1

    
    #cfiles = [CardcapFile(f) for f in cardfiles]
    #merger = CcapMerger(cardfiles)
    #fid, blk = next(merger.block_iter())

    #fileblocks = [next(f.packet_iter(nt*4*nbeam)) for f in cfiles]
    dtype = get_single_packet_dtype(nbl, True, polsum)
    fileblocks = [np.zeros((nbeam*nc*nt), dtype=dtype) for fpga in range(nfpga)]
    fb0 = fileblocks[0]
    fb0_block = fb0[:nt]
    from craco.card_averager import do_accumulate, accumulate_all
    avg = Averager(nbeam, nant, nc, nt, npol)
    niter = 10000
    do_accumulate(avg.output, avg.rescale_scales, avg.rescale_stats, avg.nant, 0,0,fb0_block, 2,6 )

    start = time.clock()
    for i in range(niter):
        do_accumulate(avg.output, avg.rescale_scales, avg.rescale_stats, avg.nant, 0,0,fb0_block, 2,6 )

    end = time.clock()
    duration = (end - start)/niter
    print(f'Do accumulate {niter} took {duration*1e6:0.1f} us')



def _main():
    test_averaging()
    
if __name__ == '__main__':
    _main()

    
    

    
    
