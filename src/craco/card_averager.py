#!/usr/bin/env python
"""
Utilities to average cardcap data from a card

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
os.environ['NUMBA_THREADING_LAYER'] = 'omp' # my TBB version complains
os.environ['NUMBA_NUM_THREADS'] = '3'
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_CPU_NAME'] = 'generic'
os.environ['NUMBA_CPU_FEATURES'] = '+sse,+sse2,+avx,+avx2,+avx512f,+avx512dq'



from craco.cardcapfile import NCHAN, NFPGA, get_indexes, NBEAM
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
    print(type(beam_data), len(beam_data), type(beam_data[0]), beam_data[0].shape, beam_data[0].dtype)

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


def vis_reshape(din, cross_idxs, dout):
    npkt, nt, nbl_in, _ = din.shape
    nbl_out = len(cross_idxs)
    expected_dout_shape = (NBEAM, nbl_out, NCHAN, nt, 2)
    assert dout.shape == expected_dout_shape, f'Invalid dout shape dout={dout.shape} expected={expected_dout_shape} {din.shape} {nbl_out} {nt}'
    assert npkt == NBEAM*NCHAN
    dout[:32, ...] = din[:32*4,...].reshape(NCHAN,32,nt,nbl_in,2)[:,:,:,cross_idxs,:].transpose(1,3,0,2,4)
    dout[32:, ...] = din[32*4:,...].reshape(NCHAN,4 ,nt,nbl_in,2)[:,:,:,cross_idxs,:].transpose(1,3,0,2,4)
    return dout
            
def average_vis_and_reshape(din, tscrunch, dout, auto_idxs, cross_idxs):
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
     
def accumulate_all2(output, rescale_scales, rescale_stats, count, nant, beam_data, valid, antenna_mask, auto_idxs, cross_idxs,  vis_fscrunch=1, vis_tscrunch=1):
    '''
    FIxed vis fscrunch
    Doesnt do CAS or ICS
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
                
class Averager:
    def __init__(self, nbeam, nant, nc, nt, npol, vis_fscrunch=6, vis_tscrunch=1,rdtype=np.float32, cdtype=np.float32, dummy_packet=None, exclude_ants=None, rescale_update_blocks=16, rescale_output_path=None):

        
        if exclude_ants is None:
            exclude_ants = []

        assert np.all(np.array(exclude_ants) - 1 < nant), f'We cant handle flagging antennas > nant={nant} We dont have the logic. execlude ants={exclude_ants}'

        self.nant_in = nant
        self.nant_out = self.nant_in - len(exclude_ants)
        nbl_with_autos = self.nant_out*(self.nant_out+1)//2
        self.nbl_with_autos = nbl_with_autos
        self.nt = nt
        self.npol = npol
        self.nc = nc
        self.vis_fscrunch = vis_fscrunch
        self.vis_tscrunch = vis_tscrunch
        self.rescale_update_blocks = rescale_update_blocks
        self.dtype = get_averaged_dtype(nbeam, self.nant_out, nc, nt, npol, vis_fscrunch, vis_tscrunch, rdtype, cdtype)
        self.output = np.zeros(nbeam, dtype=self.dtype)
        self.rescale_stats = np.zeros((nbeam, nc, self.nbl_with_autos, npol, 2), dtype=rdtype)
        self.rescale_scales = np.zeros((nbeam, nc, self.nbl_with_autos, npol, 2), dtype=rdtype)
        self.count = np.zeros(NFPGA, dtype=np.int32)
        
        _,_,self.auto_idxs,self.cross_idxs = get_indexes(self.nant_in, exclude_ants=exclude_ants)

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

        valid = np.array([pkt is not None for pkt in packets], dtype=bool)
        self.last_nvalid = valid.sum()
        data = List()
        [data.append(self.dummy_packet if pkt is None else pkt) for pkt in packets]
        log.debug('Accumulating %s', ' '.join(map(str, [d.shape for d in data])))
        for idata, d in enumerate(data):
            if d.ndim == 1:
                d.shape = (d.shape[0], 1)
            
            assert d.shape == self.dummy_packet.shape, f'Invalid shape for packet[{idata}] expected={self.dummy_packet.shape} but got {d.shape}'

            #log.info('Idata %d dhsape=%s', idata, d.shape)
            
        return self.accumulate_all(data, valid)


    def accumulate_all(self, beam_data, valid):
        '''
        Runs multi-threaded accumulation over all fpgas/coarse channels / beams / times /  baselnes
        :param: beam_data is numba List with the expected data
        '''

        use_v2 = True

        if use_v2:
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
            

        # update after first block and then every N thereafter
        if self.iblk == 0 or self.iblk % self.rescale_update_blocks == 0:
            self.update_scales()

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
