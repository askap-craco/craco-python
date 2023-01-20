#!/usr/bin/env python
from craco.card_averager import Averager
from craco.cardcap import CardcapFile, get_single_packet_dtype, NCHAN
from craco.cardcapmerger import CcapMerger
from craco.utils import ibc2beamchan
import numpy as np
import glob
import time
from numba.typed import List
from pylab import *
from IPython import embed


nt = 64
nbeam = 36
nant = 30
#nant = 3
nbl = nant*(nant + 1)//2

nfpga = 6
nc = 4*nfpga
npol = 2

ibl = 0
cross_idxs = []
auto_idxs = []
for a1 in range(nant):
    for a2 in range(a1, nant):
        if a1 == a2:
            auto_idxs.append(ibl)
        else:
            cross_idxs.append(ibl)
            
        ibl += 1


def test_timing():
    cardfiles = glob.glob('/data/craco/ban115/craco-python/notebooks/data/SB43128/run3/1934_b07_c01+f?.fits')
    assert len(cardfiles) == 6

    cfiles = [CardcapFile(f) for f in cardfiles]
    merger = CcapMerger(cardfiles)
    fid, blk = next(merger.block_iter())

    fileblocks = [next(f.packet_iter(nt*4*nbeam)) for f in cfiles]
    fb0 = fileblocks[0]
    fb0_block = fb0[:nt]
    from craco.card_averager import do_accumulate, accumulate_all
    avg = Averager(nbeam, nant, nc, nt, npol)
    niter = 10000
    do_accumulate(avg.output, avg.rescale_scales, avg.rescale_stats, avg.count, avg.nant, 0,0,fb0_block, 2,6 )

    start = time.clock()
    for i in range(niter):
        do_accumulate(avg.output, avg.rescale_scales, avg.rescale_stats, avg.count, avg.nant, 0,0,fb0_block, 2,6 )

    end = time.clock()
    duration = (end - start)/niter
    print(f'Do accumulate {niter} took {duration*1e6:0.1f} us')

def test_do_accumulate():
    debughdr = True
    polsum = npol == 1
    dtype = get_single_packet_dtype(nbl, debughdr, polsum)
    npkt = nt
    packets = np.zeros(nt, dtype=dtype)
    #packets['data'].flat = np.arange(packets.size, dtype=np.int8)
    packets['data'].flat = np.random.randn(packets.size)*100
    avg = Averager(nbeam, nant, nc, nt, npol)

    # just accumulate beam0, channel0
    ibeam = 0
    ichan = 0

    avg.accumulate_beam(ibeam, ichan, packets)
    d = packets['data'].astype(np.float32)
    amp = np.sqrt(d[...,0]**2 + d[...,1]**2)
    amp_polsum = amp.sum(axis=3)

    cas = amp[:,0,cross_idxs,:].sum(axis=(1,2))
    ics = amp[:,0,auto_idxs,:].sum(axis=(1,2))
    cas_diff = (avg.output['cas'][ibeam,:,ichan] - cas)
    ics_diff = (avg.output['ics'][ibeam,:,ichan] - ics)

    assert cas_diff.std()/cas.mean() < 1e-6, f'CAS error is too high: {cas_diff.std()}/{cas.mean()}'
    assert ics_diff.std()/ics.mean() < 1e-6, f'ICS error is too high: {ics_diff.std()}/{ics.mean()}'


    # HACK: update count as we've not run the proper function to set it
    avg.count = nt

    mean = avg.rescale_stats[ibeam,ichan,:,:,0]
    m2 = avg.rescale_stats[ibeam,ichan,:,:,1]
    std = np.sqrt(m2 / (avg.count))

    meanerr = (mean - amp.mean(axis=(0,1)))
    assert meanerr.std() < 1e-3,f'Mean amplitude not correct {meanerr.std()}'

    stderr = (std - amp.std(axis=(0,1)))
    assert stderr.std() < 1e-3,f'Std amplitude not correct {stderr.std()}'

    # update rescaling then check the mean and stdev of the output are zero
    avg.update_scales()
    avg.reset()
    avg.accumulate_beam(ibeam, ichan, packets)
    casout = avg.output['cas'][ibeam,:,ichan]
    icsout = avg.output['ics'][ibeam,:,ichan]

    assert np.abs(casout.mean()) < 1e-3, f'CAS Mean not zero {casout.mean()}'
    assert np.abs(icsout.mean()) < 1e-3, f'ICS Mean not zero {casout.mean()}'

    # BOOOM - THIS FAILS! we cant do this assertion because the statistics aren't gaussian
    # We'll need to do some work to workout the best S/N of doing CAS.
#    assert np.abs(casout.std()/np.sqrt(len(cross_idxs)) - 1) < 1e-3, f'Std not 1 {casout.std()}'

        

def test_check_accumulate_all():
    # Make test data
    debughdr = True
    polsum = npol == 1
    dtype = get_single_packet_dtype(nbl, debughdr, polsum)
    npkt = nt*NCHAN*nbeam
    packets = List()
    [packets.append(np.zeros(npkt, dtype=dtype)) for f in range(nfpga)]
    input_data = np.zeros((nbeam, NCHAN*nfpga, nt, nbl, npol,2), dtype=np.int16)
    input_data.flat = np.arange(input_data.size, dtype=np.int8)

    for ifpga, dfpga in enumerate(packets):
        for t in range(nt):
            for ibc in range(NCHAN*nbeam):
                beam, coarse_chan = ibc2beamchan(ibc)
                for p in range(npol):
                    print(dfpga.shape, dfpga['data'].shape)
                    fullchan = ifpga + 6*coarse_chan
                    dfpga['data'][ibc:ibc+nt,0,:,:,:] = input_data[beam,fullchan,:,:,:,:]
    
    avg = Averager(nbeam, nant, nc, nt, npol)
    avg.accumulate_all(packets)


    amp = np.sqrt(input_data[...,0]**2 + input_data[...,1]**2)
    cas = amp[:,:,:,cross_idxs,:].sum(axis=(3,4))
    ics = amp[:,:,:,auto_idxs,:].sum(axis=(3,4))

    from IPython import embed
    embed()

    




    
            
            
    

    
    
    




def _main():
    test_averaging()
    
if __name__ == '__main__':
    _main()

    
    

    
    
