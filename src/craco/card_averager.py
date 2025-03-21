#!/usr/bin/env python
"""
Utilities to average cardcap data from a card

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
from craco.timer import Timer
os.environ['NUMBA_THREADING_LAYER'] = 'omp' # my TBB version complains
os.environ['NUMBA_NUM_THREADS'] = '2'
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_CPU_NAME'] = 'generic'
os.environ['NUMBA_CPU_FEATURES'] = '+sse,+sse2,+avx,+avx2,+avx512f,+avx512dq'



from craco.cardcapfile import NCHAN, NFPGA, get_indexes, NBEAM, debughdr
from numba import jit,njit,prange
import numba
from numba.typed import List


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

real_dtype = numba.float32
#real_dtype = numba.int32

accumulate_locals = {'v0':real_dtype,
                     'v1':real_dtype,
                     'vsqr':real_dtype,
                     'va':real_dtype,
                    'agg_mean':real_dtype,
                     'agg_m2':real_dtype,
                     'delta':real_dtype,
                     'delta2':real_dtype}

@njit(debug=True,cache=True,fastmath=True,parallel=False, locals=accumulate_locals)
def do_accumulate(output, rescale_scales, rescale_stats, count, nant, ibeam, ichan, beam_data, antenna_mask, vis_valid, vis_fscrunch=1, vis_tscrunch=1):
    '''
    Computes ICS, CAS and averaged visibilities given a block of nt integration packets from a single FPGA
  
    makeing it parallel makes it worse
    :output: Rescaled and averaged output. 1 per beam.
    :rescale_scales: (nbeam, nbl, npol, 2) float 32 scales to apply to vis amplitudes before adding in ICS/CAS. [0] is subtracted and [1] is multiplied. 
    :rescale_stats: (nbeam, nc, nbl, npol, 2) flaot32 statistics of the visibility amplitues. [0] is the sum and [2] is the sum^2
    :count: Number of samples that have so far been used to do accumulation
    :nant: number of antennas. Should tally with number of baselines
    :ibeam: Beam number to udpate
    :ichan: Channel number to update
    :beam_data: len(nt) list containing packets
    :antenna_mask: array of bool. Include antenna_mask[iant] in CAS/ICS if sum is True. Corrs not affected (yet)
    :vis_valid: bool - set to True if you want to update the visibility data. False otherwise
    :vis_fscrunch: Visibility fscrunch factor
    :vis_tscrunch: Visibility tscrunch factor
    
    '''

    ics = output[ibeam]['ics']
    cas = output[ibeam]['cas']
    vis = output[ibeam]['vis']
    nsamp = len(beam_data)
    (nsamp2, nbl, npol, _) = beam_data[0]['data'].shape
    rs_chan_stats = rescale_stats[ibeam, ichan, ...]
    rs_chan_scales = rescale_scales[ibeam, ichan, ...]
    nt = nsamp*nsamp2

    for samp in range(nsamp):
        bd = beam_data[samp]['data']
        for samp2 in range(nsamp2):
            # continue here works for real time
            t = samp2 + nsamp2*samp
            agg_count = t + count + 1
            ochan = ichan // vis_fscrunch
            otime = t // vis_tscrunch
            a1 = 0
            a2 = 0
            output_bl = 0
            vis_bl = 0
            # looping over baseline and calculating antenna numbers is about 15% faster than
            # 2 loops over antennas
            for ibl in range(nbl):
                ants_ok = antenna_mask[a1] and antenna_mask[a2]
                #print('CAVG', ibeam, ichan, samp, samp2, ibl, a1, a2, ants_ok, vis_bl, output_bl, bd.shape, vis.shape)
                if ants_ok:
                    for pol in range(npol):
                        # continue here means we can't do real time
                        # I tested it twice and you really cant.
                        v = bd[samp2, ibl, pol, :]
                        v_real = np.float32(v[0]) # have to add np.float32 here when not using number, othewise we get nan in the sqrt
                        v_imag= np.float32(v[1])
                        # For ICS: Don't subtract before applying square  and square root.
                        vsqr = v_real*v_real + v_imag*v_imag
                        va = np.sqrt(vsqr)
                        
                        # Update mean and M2 for variance calculation using Welfords online algorithm
                        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                        agg_mean = rs_chan_stats[output_bl, pol, 0]
                        agg_m2 = rs_chan_stats[output_bl, pol, 1]
                        delta = va - agg_mean
                        agg_mean += delta / agg_count
                        delta2 = va - agg_mean
                        agg_m2 += delta * delta2

                        #print(agg_mean, output_bl, pol, delta, agg_count, rs_chan_stats.shape, va)
                        rs_chan_stats[output_bl, pol, 0] = agg_mean # Update amplitude
                        rs_chan_stats[output_bl, pol, 1] = agg_m2 # Add amplitude**2 for stdev
                        
                        offset = rs_chan_scales[output_bl, pol, 0] # offset = 0
                        scale = rs_chan_scales[output_bl, pol, 1] # scale = 1
                        va_scaled = (va + offset)*scale

                        if a1 == a2:
                            # Could just add a scaled version of v0 here, but it makes  little diffeence
                            # given there are so few autos
                            ics[t,ichan] += va_scaled
                        else:
                            cas[t,ichan] += va_scaled
                            if vis_valid:
                                vis[vis_bl, ochan, otime] += complex(v_real, v_imag)

                            if pol == npol -1 :# what a mess
                                vis_bl += 1 
                        
                    output_bl += 1
                    
                a2 += 1
                if a2 == nant:
                    a1 += 1
                    a2 = a1


@njit
def get_channel_of(chan, nc_per_fpga, fpga, nfpga):
    ichan0 = chan*nfpga + fpga
    ichan1 = (nc_per_fpga - 1 - chan)*nfpga + (nfpga - 1 - fpga) # Better but not perfect
    ichan2 = (nc_per_fpga - 1 - chan)*nfpga + (fpga) # Possibly not as good as ichan1?
    ichan3 = chan*nfpga + (nfpga - 1 - fpga) # Worse then ichan3
    ichan = ichan0
    assert 0 <= ichan < nc_per_fpga*nfpga
    
    return ichan

#@njit(parallel=True,cache=True)
def accumulate_all(output, rescale_scales, rescale_stats, count, nant, beam_data, valid, antenna_mask, vis_fscrunch=1, vis_tscrunch=1):
    nfpga= len(beam_data)
    assert nfpga == 6
    npkt = len(beam_data[0])
    nbeam, nc, nbl, npol, _ = rescale_scales.shape
    nc_per_fpga = 4
    #npkt_per_accum = npkt // (nbeam * nc_per_fpga)
    dshape = beam_data[0].shape
    #print(type(beam_data), len(beam_data), type(beam_data[0]), beam_data[0].shape, beam_data[0].dtype)

    assert len(dshape) == 2 # expected (nmsgs, npkt_per_accum)
    nmsgs = dshape[0]
    nt = output[0]['cas'].shape[0] # assume this is the same as ICS

    # Set visibility output to valid valid only if all FPGAS for this card are valid
    # this stops you getting into trouble with bright sources and only averaging some of the channels 
    # see CRACO-130
    # only worry about this is visibility frequency averaging is enabled
    if vis_fscrunch == 1:
        vis_valid = True
    else:
        vis_valid  = np.all(valid)

        
    for beam in prange(nbeam):
        for fpga in range(nfpga):
            # continue here lets us still run in real time
            #continue
            
            # TODO: Work out what should do if some data isn't valid.
            # do we Just not add it, do we note it somewhere in some arrays ... what should we do?
            isvalid = valid[fpga]
            if not isvalid:
                continue

            data = beam_data[fpga]

            for chan in range(nc_per_fpga):

                # if 36 beams, we need to work out crazy ordering
                # It there's only 1 beam, then it's just 4 channels one after the oter
                if nbeam == 36:
                    if beam < 32:
                        didx = beam + 32*chan
                    else:
                        b = beam - 32
                        didx = 32*4 + b + 4*chan
                else:
                    didx = chan


                ichan = chan*nfpga + fpga

                bd = data[didx,:]
                do_accumulate(output, rescale_scales, rescale_stats, count[fpga], nant, beam, ichan, bd, antenna_mask, vis_valid, vis_fscrunch, vis_tscrunch)
                
            count[fpga] += nt # only gets incremented if isvalid == True

@njit
def ibc2beamchan(ibc):
    '''
    Given a beam/chan index of 0 to 36beamx4chan returns the actual beam and channel
    assumes channel beam order, with beam0-31 to start, then beams 32-35 at the end
    i.e. the crazy beamformer ordering
    '''

    if ibc < 32*4:
        beam = ibc % 32
        chan = ibc // 32
    else:
        ibc2 = ibc - 32*4
        beam = ibc2 % 4 + 32
        chan = ibc2 // 4

    return (beam, chan)

@njit
def beamchan2ibc(beam:int, chan:int):
    '''
    Converts beam, channel index back to crazy beamforemer orering index
    '''
    if beam < 32:
        ibc = chan*32 + beam
    else:
        ibc = NCHAN*32 + (beam - 32) + chan*4

    return ibc
  

def average0(din):
    '''
    Average fpga and pol axes only
    '''
    dout = din['data'].mean(axis=(0,5), dtype=np.float32)
    return dout

def average1(din):
    '''
    Average the FPGA axis and the nt1,nt2 and pol axis
    '''
    #(nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    dout =  din['data'].mean(axis=(0,2,3,5), dtype=np.float32)
    return dout

def average2(din, tscrunch=4):
    '''
    Average fpga and nt1,nt2 axis with tscrunching, and pol axis
    '''
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    # This reshape takes 16 milliseconds!
    dreshape =  data.reshape(nfpga, npkt, nt1*nt2//tscrunch, tscrunch, nbl, npol, 2)
    dout = dreshape.mean(axis=(0,3,5), dtype=np.float32)
    return dout

def average3(din, tscrunch=4):
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    nttotal = nt1 * nt2
    ntout = nttotal // tscrunch
    # average first time axis
    dout =  data.mean(axis=(0,3), dtype=np.float32)

    tscrunch2 = tscrunch // nt2
    dout = dout.reshape(npkt, nt1 // tscrunch2, tscrunch2, nbl, 2).mean(axis=3)
    return dout

def average4(din, tscrunch=2, dout=None):
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    dshape = (npkt, nt1*nt2 // tscrunch, nbl, 2)
    if dout is None:
        dout = np.zeros(dshape, dtype=np.float32)
    else:
        dout[:] = 0
        
    for ifpga in range(nfpga):
        for t1 in range(nt1):
            for t2 in range(nt2):
                ttotal = t2*nt1 + t1
                tout = ttotal // tscrunch
                dout[:, tout, :, :] += data[ifpga,:,t1,t2,:,0,:]
            
    return dout


@njit(fastmath=True,debug=True, parallel=True)
def average5(data, tscrunch, dout):
    #data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    dout[...] = 0
    for ifpga in range(nfpga):
        for t1 in range(nt1):
            for t2 in range(nt2):
                ttotal = t2*nt1 + t1
                tout = ttotal // tscrunch
                dout[:, tout, :, :] += data[ifpga,:,t1,t2,:,0,:]
            
    return dout

@njit(fastmath=True,debug=True,parallel=True)
def average6(din, tscrunch, dout):
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    dout[:] = 0
    
    for ifpga in range(nfpga):
        for ipkt in range(npkt):
            for t1 in range(nt1):
                for t2 in range(nt2):
                    ttotal = t2*nt1 + t1
                    tout = ttotal // tscrunch
                    for ibl in range(nbl):
                        for c in range(nt2):
                            dout[ipkt, tout, ibl, c] += data[ifpga,ipkt,t1,t2,ibl,0,c]
                            
    n = tscrunch*nfpga
    dout *= float(1/n)
    
            
    return dout


@njit(fastmath=True,debug=True,parallel=False)
def average7(din, tscrunch, dout):
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    dout[:] = 0
    # HACK - set nbl to output NBL, which doesn't include autos, for now
    nbl = dout.shape[1]
    
    # dout shape
    # avger.output['vis'].shape
    # (36, 435, 4, 8, 2)
    
    for ifpga in range(nfpga):
        for ipkt in prange(npkt):
            beam,chan = ibc2beamchan(ipkt)
            #print(beam,chan)
            for t1 in range(nt1):
                for t2 in range(nt2):
                    ttotal = t2 + t1*nt2
                    tout = ttotal // tscrunch
                    for ibl in range(nbl):                        
                        #print(t1,t2,ttotal,tout,ibl,c, dout.shape, data.shape)
                        for c in range(2):
                            dout[beam, ibl, chan, tout, c] += data[ifpga,ipkt,t1,t2,ibl,0,c]                            

                            
    n = tscrunch*nfpga
    dout *= float(1/n)
    
            
    return dout

@njit(fastmath=True,debug=True)
def average8(din, tscrunch, dout, nant):
    '''
    Writes (beam,chan) order and removes autocorrelations
    '''
    data = din['data']
    (nfpga, npkt, nt1, nt2, nbl, _, _) = data.shape
    dout[:] = 0
    # HACK - set nbl to output NBL, which doesn't include autos, for now
    #nbl = dout.shape[1]
    
    # dout shape
    # avger.output['vis'].shape
    # (36, 435, 4, 8, 2)
    
    for ifpga in range(nfpga):
        for ipkt in range(npkt):
            beam,chan = ibc2beamchan(ipkt)
            #print(beam,chan)
            for t1 in range(nt1):
                for t2 in range(nt2):
                    ttotal = t2 + t1*nt2
                    tout = ttotal // tscrunch
                    
                    obl = 0
                    ibl = 0
                    for ia1 in range(nant):
                        for ia2 in range(ia1, nant):
                            ibl += 1
                            if ia1 == ia2:
                                continue
                                
                            for c in range(nt2):
                                #print(t1,t2,ttotal,tout,ibl,c, dout.shape, data.shape)
                                dout[beam, obl, chan, tout, c] += data[ifpga,ipkt,t1,t2,ibl,0,c]
                                
                            obl += 1
                            
                            
    n = tscrunch*nfpga
    dout *= float(1/n)
    
            
    return dout

def average9(din, tscrunch, dout):
    '''
    Try to average without doing doing ['data']
    This takes 110ms
    '''
    data = din['data'] # this line takes5 00 ns
    (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    dshape = (npkt, nt1*nt2 // tscrunch, nbl, 2)
    dout[:] = 0
    for ifpga in range(nfpga):
        dout[...] += data[ifpga, ...]

    return dout

#@njit - can't njit this - The dtype of a Buffer type cannot itself be a Buffer type,


def average10(packets, dout):
    '''
    Averaaging accross fpga, No tscrunching
    '''
    nfpga = len(packets)
    if dout is None:
        dout = np.zeros(packets[0]['data'].shape, dtype=np.float32)
    else:
        dout[:] = 0
        
    for pkt in packets:
        dout += pkt['data']

    dout *= (1./float(nfpga))
    return dout

def average11(packets, dout):
    nfpga = len(packets)
    dout[:] = 0

    for ipkt in range(len(packets)):
        pkt = packets[ipkt]
        for ibc in range(pkt.shape[0]):
            for t1 in range(pkt.shape[1]):
                dout[ibc,t1, ...] += pkt[ibc,t1]['data']

    dout *= (1./float(nfpga))

    return dout

def average12(packets, tscrunch, dout):
    nfpga = len(packets)
    dout[:] = 0
    nibc, nt1, nt2, nbl, npol, _ = packets[0]['data'].shape
    ntout = nt1 * nt2 // tscrunch

    if dout is None:
        dout = np.zeros((nibc, ntout, nbl, 2), dtype=np.float32)
        
    for pkt in packets:
        dout += pkt['data'].reshape(nibc, ntout, tscrunch, nbl, npol, 2).mean(axis=(2,4))

    dout *= (1./float(nfpga))
    return dout

def average13(packets, tscrunch, dout=None, dout2=None):
    '''
    Averages channels and tscrunch and pols
    dout has shape [144, ntout, nbl, npol, 2]

    '''
    davg = average10(packets,dout=dout2)
    (nibc, nt1, nt2, nbl, npol, _) = packets[0]['data'].shape
    ntout = nt1*nt2//tscrunch
    if dout is None:
        dout = np.empty((nibc,ntout,nbl,2), dtype=np.float32)

    dout[...] = davg.reshape(NBEAM*NCHAN,ntout,tscrunch,nbl,npol,2).mean(axis=(2,4))
    return dout


def get_flat_dtype(packet_dtype):
    '''
    Returns the dtype for a packet where the 'data' portion has had the last 3 dimensions
    (i.e. nbl, npol, ncomplex=2) flattened

    '''
    flat_dtype = debughdr[:]
    (nt2, nbl, npol, ncomp) = packet_dtype.shape
    flat_dtype.append(('data','<i2', (nt2, nbl*npol*ncomp)))
    flat_dtype = np.dtype(flat_dtype)
    return flat_dtype

def flatten_packets(packets):
    '''
    Returns a numba list of packets. each packet is a view of the original packet 
    with the dtype set to have the last 3 dimensions flattned
    @see get_flat_dtype
    
    '''
    flat_dtype = get_flat_dtype(packets[0].dtype['data'])
    packet_list_flat = List()
    [packet_list_flat.append(pkt.view(flat_dtype)) for pkt in packets]
    return packet_list_flat

@njit(fastmath=True, boundscheck=True, parallel=False)
def average15(packets_flat, valid, cross_idxs, output, tscrunch, scratch):
    '''
    input: packets with flattened few axes - see flatten_packets()

    Fscrunch by 6
    ncout = NCHAN
    ntout = nt1*nt2 // tscrunch
    scratch = np.zeros((ncout, ntout, nprod), dtype=np.float32)
         _,nt1,nt2, nprod = packets_flat[0]['data'].shape
    '''
    nfpga = len(packets_flat)
    vis = output['vis']
    #print(packets_flat[0]['data'].shape, vis.shape)
    nbeam,ncross, nchan, nt, ncomp = vis.shape
    _,nt1,nt2, nprod = packets_flat[0]['data'].shape
    ncout = NCHAN
    ntout = nt1*nt2 // tscrunch
    scale = np.float32(1./(tscrunch*nfpga))
    
    #nprod), dtype=np.float32)if scratch is None:
    #    scratch = np.zeros((ncout, ntout, 
            
    for ibeam in prange(NBEAM):
        scratch[:] = 0

        # calculate average in scratch area - do it so its easily vectorized
        for ifpga in range(nfpga): # doing FPGA 
            for ichan in range(ncout):
                ibc = beamchan2ibc(ibeam, ichan)
                for it1 in range(nt1):
                    d = packets_flat[ifpga][ibc,it1]['data']
                    for it2 in range(nt2):
                        tidx = it2 + nt2*it1
                        toutidx = tidx // tscrunch
                        scratch[ichan, toutidx, :] += d[it2,:]

        # transpose data and select only cross correlations. apply scale to bring it back to an average
        visout = output[ibeam]['vis']
        for ibl in range(len(cross_idxs)):
            blidx = cross_idxs[ibl]
            assert ibl < visout.shape[0]

            for ichan in range(ncout):
                for it in range(ntout):
                    start = blidx*2
                    end = start + 2
                    #print(output[ibeam]['vis'].shape, scratch.shape, ichan, it, ibl, start, end)
                    # If you do an assignment with the slice and multiply by the scale it slows to a grinding halt. Don't do it.
                    visout[ibl, ichan,  it, 0] = scratch[ichan, it, start]*scale
                    visout[ibl, ichan,  it, 1] = scratch[ichan, it, start+1]*scale
     

    return (output, scratch)

def packet_data_reshape(packets, dout=None):
    '''
    Reshape a list of packets into a numpy array
    Packets have a weird shape due to funny beamformer ordering see ibc2beamchan
    And chanenls are mixed up to to FPGA ordering
    returned shape = (NBEAM, nfpga*NCHAN, nt1*nt2, nbl, npol, ncomp)
    '''
    data = np.array(packets)['data']
    (nfpga, nibc, nt1, nt2, nbl, npol, ncomp) = data.shape 

    ncout = nfpga*NCHAN
    ntout = nt1*nt2
    dout_shape = (NBEAM, ncout, ntout, nbl, npol, ncomp)
    if dout is None:
        dout = np.zeros(dout_shape, data.dtype)

    assert dout.shape == dout_shape

    def doreshape(nbeam, d):
        return d.reshape(nfpga,NCHAN,nbeam,nt1*nt2,nbl,npol,ncomp) \
                .transpose(2,1,0,3,4,5,6) \
                .reshape(nbeam,nfpga*NCHAN,nt1*nt2,nbl,npol,ncomp)
    dout[:32, ...] = doreshape(32, data[:,:32*4,...])
    dout[32:, ...] = doreshape(4, data[:, 32*4:,...])
    return dout

def scrunch_vis(vis, fscrunch, tscrunch):
    '''
    Tscrunch and fscrunch visibility data made by packet_data-reshape
    used mostly for testing    
    casts to float32
    '''
    (nbeam, nc, nt, nbl, npol, ncomp) = vis.shape
    dout = vis.reshape(nbeam, nc//fscrunch, fscrunch, nt // tscrunch, tscrunch, nbl, npol, ncomp) \
        .astype(np.float32).mean(axis=(2,4))
    return dout

def vis_reshape(din, cross_idxs, dout=None):
    npkt, nt, nbl_in, _ = din.shape
    nbl_out = len(cross_idxs)
    expected_dout_shape = (NBEAM, nbl_out, NCHAN, nt, 2)
    if dout is None:
        dout = np.zeros(expected_dout_shape, dtype=din.dtype)
    assert dout.shape == expected_dout_shape, f'Invalid dout shape dout={dout.shape} expected={expected_dout_shape} {din.shape} {nbl_out} {nt}'
    assert npkt == NBEAM*NCHAN
    dout[:32, ...] = din[:32*4,...].reshape(NCHAN,32,nt,nbl_in,2)[:,:,:,cross_idxs,:].transpose(1,3,0,2,4)
    dout[32:, ...] = din[32*4:,...].reshape(NCHAN,4 ,nt,nbl_in,2)[:,:,:,cross_idxs,:].transpose(1,3,0,2,4)
    return dout
            
def average_vis_and_reshape(din, tscrunch, dout, cross_idxs):
    '''
    Writes (beam,chan) order and removes autocorrelations
    Fixed fscrunch at 6
    
    '''
    data = din['data']

    (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    assert npol == 1, 'Expect npol = 1 in this thing'
    
    dout[:] = 0
    # HACK - set nbl to output NBL, which doesn't include autos, for now
    #nbl = dout.shape[1]
    ntout = nt1*nt2//tscrunch
    nblout = len(cross_idxs)
    
    d = average2(din,tscrunch)

    vis_reshape(d, cross_idxs, dout)
            
    return dout

def average_pkts_and_reshape(packets, tscrunch, dout, cross_idxs, d1=None, d2=None):
    avg = average13(packets, tscrunch)
    vis_reshape(avg, cross_idxs, dout)
    return dout

def calc_ics(data, auto_idxs):
    '''
    Does no rescaling. Just average over real parts
    Does pol sum and ICS
    :data: = (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape or an iterable length(nfpga) with shape (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    :auto_idxs: list of the autocorrelation indexs
    :returns: ndarray output shape is (6,144,nt1,nt2) 
    '''
    # infuraiatignly, occasionally when you index by integers the axis comes out the front
    if isinstance(data, np.ndarray):
        dmean = data[:,:,:,:,auto_idxs,:,0].mean(axis=(0,5), dtype=np.float32)
    else:
        dmean = np.array([d['data'][:,:,:,auto_idxs,:,0].mean(axis=(0,-1), dtype=np.float32) for d in data])
    
    return dmean

def calc_and_reshape_ics(data, auto_idxs, valid, output):
    '''
    :data: = (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape or an iterable length(nfpga) with shape (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    :returns: ICS output shape = (36, 32, 24) = (nbeam,ntime,nchan)
    '''
    if isinstance(data, np.ndarray):
        (nfpga, npkt, nt1, nt2, nbl, npol, _) = data.shape
    else:
        nfpga = len(data)
        npkt, nt1, nt2, nbl, npol, _ = data[0]['data'].shape
        
    nttotal = nt1*nt2
    dmean = calc_ics(data, auto_idxs)
    assert dmean.shape == (nfpga, NCHAN*NBEAM, nt1, nt2)
    #print(data.shape,output.shape,dmean.shape)
    # ICS output shape = (36, 32, 24) = (nbeam,ntime,nchan)
    # data shape as above
    # dmean shape is (6,144,16,2) = (nfpga,nbeam*nchan,nt1,nt2)
    
    # flag FPGAS that are not valid
    dmean[~valid,...] = 0

    output[:32,...] = dmean[:,:32*4 ,:,:].reshape(nfpga,NCHAN,32,nttotal).transpose(2,3,1,0).reshape(32,nttotal,NCHAN*nfpga)
    output[32:,...] = dmean[:, 32*4:,:,:].reshape(nfpga,NCHAN,4 ,nttotal).transpose(2,3,1,0).reshape(4,nttotal,NCHAN*nfpga)
    return output

@njit
def calc_ics1(packets_flat, valid, auto_idxs, output):
    '''
    Better version of calc_ics = does all the good stuff in 1ms.
    '''
    nfpga = len(packets_flat)
    vis = output['vis']
    #print(packets_flat[0]['data'].shape, vis.shape)
    nbeam,ncross, nchan, nt, ncomp = vis.shape
    _,nt1,nt2, nprod = packets_flat[0]['data'].shape
    nant = len(auto_idxs)
    scale = np.float32(1/nant)
    
    for ibeam in prange(NBEAM):
        # calculate ICS
        output[ibeam]['ics'][:] = 0

        for ic in range(NCHAN):
            ibc = beamchan2ibc(ibeam, ic)
            for ifpga in range(nfpga):
                cout =  ifpga + nfpga * ic                        
                for it1 in range(nt1):
                    d = packets_flat[ifpga][ibc,it1]['data']
                    for it2 in range(nt2):
                        s = np.float32(0)
                        tout = it2 + nt2*it1
                        for ibl in range(len(auto_idxs)):
                            blidx = auto_idxs[ibl]                            
                            idx = blidx*2 # find real part only of value were every second part is real
                            s += d[it2,idx]
                        
                        output[ibeam]['ics'][tout, cout] = s*scale
    return output
     
def accumulate_all2(output, rescale_scales, rescale_stats, count, nant, beam_data, valid, antenna_mask, auto_idxs, cross_idxs,  vis_fscrunch=1, vis_tscrunch=1):
    '''
    FIxed vis fscrunch
    Doesnt do CAS 
    '''
    assert vis_fscrunch == 6

    # we average all FPGAs into all 4 channels. If any FPGA is not valid, then all 4 channels will be under-cooked
    # Therefore, if any FPGA is flagged, the entire visibility-summed data is flagged
    if np.all(valid):
        # this takes about 70m,s on athena. Which seems like an awefully bloody long time
        average_pkts_and_reshape(beam_data, vis_tscrunch, output['vis'],  cross_idxs)
    else:
        output['vis'] = 0

    # doint calc and rescape ics adds abotu 20mn to this call
    calc_and_reshape_ics(beam_data, auto_idxs, valid, output['ics'])
    
    return output

def accumulate_all3(output, rescale_scales, rescale_stats, count, nant, \
                     beam_data, valid, antenna_mask, auto_idxs, cross_idxs, scratch,
                     vis_fscrunch=1, vis_tscrunch=1):
    '''
    FIxed vis fscrunch
    Doesnt do CAS 
    '''
    t = Timer()
    assert vis_fscrunch == 6
    packets_flat = flatten_packets(beam_data)
    t.tick('Flatten packets')


    # we average all FPGAs into all 4 channels. If any FPGA is not valid, then all 4 channels will be under-cooked
    # Therefore, if any FPGA is flagged, the entire visibility-summed data is flagged
    if np.all(valid):
        # this takes 30ms and has no memory allocation. See the numba averaging
        # evaluation on athena
        average15(packets_flat, valid, cross_idxs, output, vis_tscrunch, scratch)
        t.tick('average15')
    else:
        output['vis'] = 0
        t.tick('Flag0')

    # doint calc and rescape ics adds abotu 20mn to this call
    calc_ics1(packets_flat, valid, auto_idxs, output)
    t.tick('calc_ics1')
    
    return output



def get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, rdtype=np.float32, cdtype=np.float32):

    nbl = nant*(nant-1)//2
    vis_nt = nt // vis_tscrunch
    vis_nc = nc // vis_fscrunch

    assert nt % vis_tscrunch == 0, f'Tscrunch should divide into nt. nt={nt} tscrunch={vis_tscrunch}'
    assert nc % vis_fscrunch == 0, f'Fscrunch should divide into nc. nc={nc} fscrunch={vis_fscrunch}'
    assert nbeam > 0
    assert nant > 0
    assert nc > 0
    assert nt > 0
    assert vis_fscrunch >= 0
    assert vis_tscrunch >= 0
    assert vis_nt > 0
    assert vis_nc > 0
    assert rdtype in (np.float32, np.int16)
    
    if cdtype == np.complex64:
        vishape = (nbl, vis_nc, vis_nt)
    else: # assumed real type
        vishape = (nbl, vis_nc, vis_nt, 2)

    dt = np.dtype([('ics', rdtype, (nt, nc)),
                   ('cas', rdtype, (nt, nc)),
                   ('vis', cdtype, vishape)])

    assert dt.itemsize > 0, f'Averaged dtype is equal to zero dt={dt}  vishape={vishape}'

    return dt

def packets_to_data(packets, dummy_packet):
    '''
    Converts the List of packets potentially containing None
    To a numba list of packets, with the None's replaced by the dummy packet
    And a numpy bool array (valid) which is Ture where the packet was not None
    '''
    t = Timer()
    valid = np.array([pkt is not None for pkt in packets], dtype=bool)
    t.tick('make valid')

    data = List()
    t.tick('make list')
    [data.append(dummy_packet if pkt is None else pkt) for pkt in packets]
    t.tick('append list')
    for idata, d in enumerate(data):
        if d.ndim == 1:
            d.shape = (d.shape[0], 1)
        
        assert d.shape == dummy_packet.shape, f'Invalid shape for packet[{idata}] expected={dummy_packet.shape} but got {d.shape}'

        #log.info('Idata %d dhsape=%s', idata, d.shape)
        
    t.tick('Do reshape')
    return (data, valid)
                
class Averager:
    def __init__(self, nbeam, nant, nc, nt, npol, 
                 vis_fscrunch=6, vis_tscrunch=1,
                 rdtype=np.float32, cdtype=np.float32, 
                 dummy_packet=None, exclude_ants=None, 
                 rescale_update_blocks=16, rescale_output_path=None,
                 version=3):

        #numba.set_num_threads(2)

        if exclude_ants is None:
            exclude_ants = []

        assert np.all(np.array(exclude_ants) - 1 < nant), f'We cant handle flagging antennas > nant={nant} We dont have the logic. execlude ants={exclude_ants}'

        self.nant_in = nant
        self.nant_out = self.nant_in - len(exclude_ants)
        nbl_with_autos = self.nant_out*(self.nant_out+1)//2
        self.nbl_in_with_autos = nant*(nant+1)//2
        self.nbl_with_autos = nbl_with_autos
        self.nt = nt
        self.npol = npol
        self.nc = nc
        self.vis_fscrunch = vis_fscrunch
        self.vis_tscrunch = vis_tscrunch
        self.version = version
        self.rescale_update_blocks = rescale_update_blocks
        self.dtype = get_averaged_dtype(nbeam, self.nant_out, nc, nt, npol, vis_fscrunch, vis_tscrunch, rdtype, cdtype)
        self.output = np.zeros(nbeam, dtype=self.dtype)
        self.rescale_stats = np.zeros((nbeam, nc, self.nbl_with_autos, npol, 2), dtype=rdtype)
        self.rescale_scales = np.zeros((nbeam, nc, self.nbl_with_autos, npol, 2), dtype=rdtype)
        self.count = np.zeros(NFPGA, dtype=np.int32)
        
        _,_,self.auto_idxs,self.cross_idxs = get_indexes(self.nant_in, exclude_ants=exclude_ants)
        # need to be numpy arrays for numba
        self.auto_idxs = np.array(self.auto_idxs)
        self.cross_idxs = np.array(self.cross_idxs)

        self.scratch = np.zeros((NCHAN, nt // vis_tscrunch, self.nbl_in_with_autos*2*self.npol), dtype=np.float32)

        assert self.output[0]['cas'].shape == self.output[0]['ics'].shape, f"do_accumulate assumes cas and ICS work on same shape. CAS shape={self.output[0]['cas'].shape} ICS shape={self.output[0]['ics'].shape}"


        self.exclude_ants = set(map(int, exclude_ants))
        self.antenna_mask = np.array([False if (iant+1) in exclude_ants else True for iant in range(self.nant_in)])
        log.info('There are %s valid antennas in mask=%s. Excluding=%s',
                 sum(self.antenna_mask==True), self.antenna_mask, self.exclude_ants)
        assert len(self.antenna_mask) == self.nant_in
        assert not np.all(self.antenna_mask==False), 'All antennas were masked'

        self.rescale_output_path = rescale_output_path
        if self.rescale_output_path is not None:
            os.makedirs(self.rescale_output_path, exist_ok=True)


        # OMG - I the fact that dummy_packet has to come in tells you that
        # this is all wrong. I need to do some tidying up
        self.iblk = 0        
        if dummy_packet is not None:
            self.dummy_packet = dummy_packet
            
            # run it so it numba compiles it
            packets = [self.dummy_packet for i in range(NFPGA)]

            self.accumulate_packets(packets)
            self.iblk = 0

        self.reset()
        self.reset_scales()

        # allocate temporary buffers - this is hacky as we shouldnt need them but it looks like
        # numpy is faster than numba for this stuff

    def reset(self):
        self.output[:] = 0

    def reset_scales(self):
        self.rescale_scales[...,0] = 0 # offset = 0
        self.rescale_scales[...,1] = 1 # scale = 1
        self.count[:] = 0 # count = 0

    def update_scales(self):
        '''
        Updates the rescale scale values - converts sum and sum^2 into mean and varaiance 
        And sets rescale scales to subtract mean and divide by standard deviation

        :see: Welfords algorithm https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        '''

        mean = self.rescale_stats[...,0]
        m2 = self.rescale_stats[...,1]

        # this is a bit yucky and hacky
        # count is the number of valid samples we've had, per FPGA. We need to make
        # a useful array out of it we can use in the variance
        count = np.zeros((1, self.nc, 1, 1), dtype=np.int32)
        # populate count array assuming craczy FPGA order
        for ifpga in range(NFPGA):
            count[0,ifpga::NFPGA,0,0] = self.count[ifpga]

        variance = m2 / count #ill produce NAN where count=0
        # crappy way of doing it as itwill mak ewarnigns and its slow
        variance[np.isnan(variance)] = 0

        #sample_variance = m2 / (self.count - 1)
        # not sure if I should use variance, or sample variance, let's use variance

        stdev = np.sqrt(variance)
        offset = -mean
        scale = 1/stdev
        scale[np.isinf(scale)] = 0

        if self.rescale_output_path is not None:
            fout = os.path.join(self.rescale_output_path, f'rescale_{self.iblk:03d}')
            np.savez(fout, mean=mean,count=count,scount=self.count,variance=variance,stdev=stdev,offset=offset,scale=scale)


        self.rescale_scales[...,0] = offset
        self.rescale_scales[...,1] = scale
        
        # reset stats
        self.rescale_stats[:] = 0
        self.count[:] = 0

    def accumulate_packets(self, packets):
        '''
        Converst packets to beam datda and runs accumulate_all
        '''
                #print('RX', ibuf, fids, type(packets), len(packets),  type(packets[0]), len(packets[0]), type(packets[0][0]), packets[0][0], type(packets[0][1]))
        # Ugh, this is ugly, packets is a list of (fid, data) = need to tidy this up
        # data = List() # construct a typed list for NUMBA - not sure if this needs to be cached  if it's slow

        # also, if a packet is missing the iterator returns None, but Numba List() doesn't like None.

 
        data, valid = packets_to_data(packets, self.dummy_packet)
        return self.accumulate_all(data, valid)
    

    def reference_average(self, packets):
        '''
        Calculate the reference accumulation of given packets
        given the data
        Returns an array - same shape as self.output but leaves the actual self.output alone
        '''
        vis = packet_data_reshape(packets)
        auto_idxs = self.auto_idxs
        cross_idxs = self.cross_idxs
        ics = vis[:,:,:,auto_idxs,0,0].mean(axis=3).transpose(0,2,1)
        cas2 = (vis[:,:,:,cross_idxs,0,0]**2 + vis[:,:,:,cross_idxs,0,1]**2).mean(axis=3)
        cas2 = cas2.transpose(0,2,1)
        # make viss (nbeam, nchan, nt, nbl, 2)
        viss = scrunch_vis(vis, self.vis_fscrunch, \
                           self.vis_tscrunch)[:,:,:,:,0,:].transpose(0,3,1,2,4)
        
        doutput = self.output.copy()
        
        doutput['vis'] = viss[:,cross_idxs,...]
        doutput['ics'] = ics
        doutput['cas'] = cas2
        return doutput
    

    def accumulate_all(self, beam_data, valid):
        '''
        Runs multi-threaded accumulation over all fpgas/coarse channels / beams / times /  baselnes
        :param: beam_data is numba List with the expected data
        '''

        version = self.version
    
        t = Timer()
        self.last_nvalid = valid.sum()
        t.tick('sum nvalid')
        if version == 3:
            accumulate_all3(self.output,
                        self.rescale_scales,
                        self.rescale_stats,
                        self.count,
                        self.nant_in,
                        beam_data, # np.array(beam_data) takes 22ms
                        valid,
                        self.antenna_mask,
                        self.auto_idxs,
                        self.cross_idxs,
                        self.scratch,
                        self.vis_fscrunch,
                        self.vis_tscrunch)

        elif version == 2:
            #accuulate all 2 doesnt do rescaling
            # so dn't need to reset
            #self.reset()
            accumulate_all2(self.output,
                        self.rescale_scales,
                        self.rescale_stats,
                        self.count,
                        self.nant_in,
                        beam_data, # np.array(beam_data) takes 22ms
                        valid,
                        self.antenna_mask,
                        self.auto_idxs,
                        self.cross_idxs,
                        self.vis_fscrunch,
                        self.vis_tscrunch)
        else:
            self.reset() 
            accumulate_all(self.output,
                        self.rescale_scales,
                        self.rescale_stats,
                        self.count,
                        self.nant_in,
                        beam_data,
                        valid,
                        self.antenna_mask,
                        self.vis_fscrunch,
                        self.vis_tscrunch)

        t.tick('Run accumulate')            

        # update after first block and then every N thereafter
        if self.iblk == 0 or self.iblk % self.rescale_update_blocks == 0:
            self.update_scales()
            t.tick('Update scales')

        self.iblk += 1

        return self.output

    def accumulate_beam(self, ibeam, ichan, beam_data, ifpga=0, antmask=None, vis_valid=True):
        if antmask is None:
            antmask = np.ones(self.nant_in, dtype=bool)
        
        do_accumulate(self.output, self.rescale_scales, self.rescale_stats, self.count[ifpga], self.nant_in, ibeam, ichan, beam_data, antmask, vis_valid, self.vis_fscrunch, self.vis_tscrunch)

        return self.output


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
