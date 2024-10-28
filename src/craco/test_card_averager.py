#!/usr/bin/env python
from craco.card_averager import *
from craco.cardcap import CardcapFile, get_single_packet_dtype, NCHAN
from craco.cardcapmerger import CcapMerger
from craco.timer import Timer
import numpy as np
import glob
import time
from numba.typed import List
from pylab import *
from IPython import embed
from pytest import fixture

nt = 64
nbeam = 36
nant = 30
#nant = 3
nbl = nant*(nant + 1)//2

nfpga = 6
nc = 4*nfpga
npol = 1

_,_,auto_idxs,cross_idxs = get_indexes(nant)

def test_timing():
    #cardfiles = glob.glob('/data/craco/ban115/craco-python/notebooks/data/SB43128/run3/1934_b07_c01+f?.fits')
    #assert len(cardfiles) == 6
    #fileblocks = [next(f.packet_iter(nt*4*nbeam)) for f in cfiles]
    polsum = True
    dtype = get_single_packet_dtype(nbl, True, polsum)
    fileblocks = [np.zeros((nbeam*nc*nt), dtype=dtype) for fpga in range(nfpga)]
    fb0 = fileblocks[0]
    fb0_block = fb0[:nt]
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
                    #print(dfpga.shape, dfpga['data'].shape)
                    fullchan = ifpga + 6*coarse_chan
                    dfpga['data'][ibc:ibc+nt,0,:,:,:] = input_data[beam,fullchan,:,:,:,:]

    valid = np.ones(nfpga, dtype=bool)
    
    avg = Averager(nbeam, nant, nc, nt, npol)
    avg.accumulate_all(packets, valid)


    amp = np.sqrt(input_data[...,0]**2 + input_data[...,1]**2)
    cas = amp[:,:,:,cross_idxs,:].sum(axis=(3,4))
    ics = amp[:,:,:,auto_idxs,:].sum(axis=(3,4))

    from IPython import embed
    embed()

@fixture
def packets():
    nfpga = 6    
    nt = 32
    npkt = NBEAM*NCHAN
    pktshape = (npkt, nt)
    polsum = True
    debughdr = True
    dtype = get_single_packet_dtype(nbl, debughdr, polsum)
    din_list = [np.zeros(pktshape, dtype=dtype) for i in range(nfpga)]

    for d in din_list:
        d['data'] = (np.random.rand(*d['data'].shape)-0.5)*32000
        d['data'][:,:,:,auto_idxs,:,1] = 0 # autos have 0 imaginary part
        assert not np.all(d['data'] == 0)


    pkts = [pkt for pkt in din_list]
    packet_list = List()
    [packet_list.append(pkt) for pkt in pkts]
    return packet_list


def test_packet_data_reshape(packets):
    pd = packet_data_reshape(packets) # shape (NBEAM, nc, NT, nbl, npol, ncomp)

    # packet shape [nfpga]['data'][nibc,nt1,nt2,nbl,npol,nc]
    # last 3 dimensions all equal = nbl, npol, nc
    nfpga = len(packets)
    (nibc, nt1, nt2, nbl, npol, ncomp) = packets[0]['data'].shape

    # spot tests
    assert np.all(packets[0]['data'][0,0,0,...] == pd[0,0,0,...])

    for ibc in range(nibc):
        for ifpga in range(nfpga):
            for it1 in range(nt1):
                for it2 in range(nt2):
                    (beam, chan) = ibc2beamchan(ibc)
                    tchan = ifpga + nfpga*chan
                    t = it2 + nt2*it1
                    assert np.all(pd[beam, tchan, t, ...] == packets[ifpga]['data'][ibc,it1,it2,...] )    

    assert np.all(pd[:,:,:,auto_idxs,:,1] == 0), 'Autos should have no imaginary part'

def test_scrunch_vis(packets):
    vis = packet_data_reshape(packets) # shape (NBEAM, nc, NT, nbl, npol, ncomp)
    tscrunch = 4
    fscrunch = 6
    vss = scrunch_vis(vis, fscrunch, tscrunch)
    
    # check first channel and integration
    np.testing.assert_allclose(vss[:,0,0,:,:,:], vis[:,:fscrunch,:tscrunch,...].astype(float).mean(axis=(1,2)))

def test_averager_accumulate_same_v2_v3(packets):   
    tscrunch = 4
    fscrunch = 6
    
    beam_data = packets
                
    valid = np.ones(len(packets), dtype=bool)

    avger2 = Averager(nbeam,nant,nc,nt,npol,fscrunch,tscrunch, dummy_packet=packets[0], version=2)       
    avger3 = Averager(nbeam,nant,nc,nt,npol,fscrunch,tscrunch, dummy_packet=packets[0], version=3)

    avger2.accumulate_packets(packets)
    avger3.accumulate_packets(packets)

    np.testing.assert_allclose(avger2.output['vis'], avger3.output['vis'], rtol=1e-6)
    np.testing.assert_allclose(avger2.output['ics'], avger3.output['ics'], rtol=1e-6)
    np.testing.assert_allclose(avger2.output['cas'], avger3.output['cas'], rtol=1e-6)


def test_averager_accumulate_packets_correct(packets):   
    tscrunch = 4
    fscrunch = 6
        
    avger = Averager(nbeam,nant,nc,nt,npol,fscrunch,tscrunch, dummy_packet=packets[0])
    expected = avger.reference_average(packets)
    avger.accumulate_packets(packets)
    np.testing.assert_allclose(avger.output['vis'], expected['vis'], rtol=1e-6)
    np.testing.assert_allclose(avger.output['ics'], expected['ics'], rtol=1e-6)

def test_averager_accumulate_packets_timing(packets):   
    niter = 10
    tscrunch = 1
    fscrunch = 6
        
    avger = Averager(nbeam,nant,nc,nt,npol,fscrunch,tscrunch, dummy_packet=packets[0])
    avger.accumulate_packets(packets)
    t = Timer()
    for i in range(niter):
        avger.accumulate_packets(packets)
    t.tick('Averaging)')
    print(f'Do accumulate {t}')


def test_ibc2beamchan():
    assert ibc2beamchan(0) == (0,0)
    assert ibc2beamchan(1) == (1,0)
    assert ibc2beamchan(2) == (2,0)
    assert ibc2beamchan(3) == (3,0)
    assert ibc2beamchan(31) == (31,0)
    assert ibc2beamchan(32) == (0,1)
    assert ibc2beamchan(32+1) == (1,1)
    assert ibc2beamchan(4*32-1) == (31,3)
    assert ibc2beamchan(4*32+0) == (32,0)
    assert ibc2beamchan(4*32+1) == (33,0)
    assert ibc2beamchan(4*32+2) == (34,0)
    assert ibc2beamchan(4*32+3) == (35,0)
    assert ibc2beamchan(4*32+4) == (32,1)
    assert ibc2beamchan(4*32+5) == (33,1)
    assert ibc2beamchan(4*32+6) == (34,1)
    assert ibc2beamchan(4*32+7) == (35,1)

def test_beamchan2ibc():
    for ibc in range(NBEAM*NCHAN):
        bc = ibc2beamchan(ibc)
        assert beamchan2ibc(*bc) == ibc, 'Invlid {ibc}. Expected {bc}'



def _main():
    test_averaging()
    
if __name__ == '__main__':
    _main()

    
    

    
    
