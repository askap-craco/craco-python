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
os.environ['NUMBA_NUM_THREADS'] = '4'

from craco.cardcap import NCHAN, NFPGA
from numba import jit,njit,prange
import numba
from numba.typed import List


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

real_dtype = numba.float32
#real_dtype = numba.int32

@njit(debug=True,fastmath=True,parallel=False, locals={'v0':real_dtype,
                                                       'v1':real_dtype,
                                                       'vsqr':real_dtype,
                                                       'va':real_dtype,
                                                       'agg_mean':real_dtype,
                                                      'agg_m2':real_dtype,
                                                      'delta':real_dtype,
                                                       'delta2':real_dtype})
def do_accumulate(output, rescale_scales, rescale_stats, count, nant, ibeam, ichan, beam_data, antenna_mask, vis_fscrunch=1, vis_tscrunch=1):
    '''
    Computes ICS, CAS and averaged visibilities given a block of nt integration packets from a single FPGA
    
    makeing it parallel makes it worse
    :output: Rescaled and averaged output. 1 per beam.
    :rescale_scales: (nbeam, nbl, npol, 2) float 32 scales to apply to vis amplitudes before adding in ICS/CAS. [0] is subtracted and [1] is multiplied. 
    :rescale_stats: (nbeam, nc, nbl, npol, 2) flaot32 statistics of the visibility amplitues. [0] is the sum and [2] is the sum^2
    :count: Number of samples that have so far been used to do accumulation
    :nant: number of antnenas. Should tally with number of baselines
    :ibeam: Beam number to udpate
    :ichan: Channel number to update
    :beam_data: len(nt) list containing packets
    :antenna_mask: array of bool. Include antenna_mask[iant] in CAS/ICS if sum is True. Corrs not affected (yet)
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
            t = samp2 + nsamp2*samp
            agg_count = t + count + 1
            ochan = ichan // vis_fscrunch
            otime = t // vis_tscrunch
            a1 = 0
            a2 = 0
            # looping over baseline and calculating antenna numbers is about 15% faster than
            # 2 loops over antennas
            for ibl in range(nbl):
                for pol in range(npol):
                    v = bd[samp2, ibl, pol, :]
                    v_real = np.float32(v[0]) # have to add np.float32 here when not using number, othewise we get nan in the sqrt
                    v_imag= np.float32(v[1])
                    # For ICS: Don't subtract before applying square  and square root.
                    vsqr = v_real*v_real + v_imag*v_imag
                    va = np.sqrt(vsqr)

                    # Update mean and M2 for variance calculation using Welfords online algorithm
                    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                    agg_mean = rs_chan_stats[ibl, pol, 0]
                    agg_m2 = rs_chan_stats[ibl, pol, 1]
                    delta = va - agg_mean
                    agg_mean += delta / agg_count
                    delta2 = va - agg_mean
                    agg_m2 += delta * delta2
                    rs_chan_stats[ibl, pol, 0] = agg_mean # Update amplitude
                    rs_chan_stats[ibl, pol, 1] = agg_m2 # Add amplitude**2 for stdev
                    
                    offset = rs_chan_scales[ibl, pol, 0] # offset = 0
                    scale = rs_chan_scales[ibl, pol, 1] # scale = 1
                    va_scaled = (va + offset)*scale

                    ants_ok = antenna_mask[a1] and antenna_mask[a2]

                    if ants_ok: # only affects CAS/ICS currently. Willl need to resize vis otherwise
                        if a1 == a2:
                            # Could just add a scaled version of v0 here, but it makes  little diffeence
                            # given there are so few autos
                            ics[t,ichan] += va_scaled
                        else:
                            cas[t,ichan] += va_scaled

                    vis[ibl, ochan, otime] += complex(v_real, v_imag)
                    
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

@njit(parallel=True)
def accumulate_all(output, rescale_scales, rescale_stats, count, nant, beam_data, valid, antenna_mask, vis_fscrunch=1, vis_tscrunch=1):
    nfpga= len(beam_data)
    assert nfpga == 6
    npkt = len(beam_data[0])
    nbeam, nc, nbl, npol, _ = rescale_scales.shape
    nc_per_fpga = 4
    #npkt_per_accum = npkt // (nbeam * nc_per_fpga)
    dshape = beam_data[0].shape
    assert len(dshape) == 2 # expected (nmsgs, npkt_per_accum)
    nmsgs = dshape[0]
    npkt_per_accum = dshape[1]
    nt = output[0]['cas'].shape[0] # assume this is the same as ICS
    for beam in prange(nbeam):
        for fpga in range(nfpga):
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
                do_accumulate(output, rescale_scales, rescale_stats, count[fpga], nant, beam, ichan, bd, antenna_mask,vis_fscrunch, vis_tscrunch)
                
            count[fpga] += nt # only gets incremented if isvalid == True


def get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, rdtype=np.float32, cdtype=np.complex64):

    nbl = nant*(nant+1)//2
    assert nt % vis_tscrunch == 0, 'Tscrunch should divide into nt'
    assert nc % vis_fscrunch == 0, 'Fscrunch should divide into nc'
    vis_nt = nt // vis_tscrunch
    vis_nc = nc // vis_fscrunch
    if cdtype == np.complex64:
        vishape = (nbl, vis_nc, vis_nt)
    else: # assumed real type
        vishape = (nbl, vis_nc, vis_nt, 2)

    dt = np.dtype([('ics', rdtype, (nt, nc)),
                   ('cas', rdtype, (nt, nc)),
                   ('vis', cdtype, vishape)])

    return dt
                
class Averager:
    def __init__(self, nbeam, nant, nc, nt, npol, vis_fscrunch=6, vis_tscrunch=1,rdtype=np.float32, cdtype=np.complex64, dummy_packet=None, exclude_ants=[], rescale_update_blocks=16, rescale_output_path=None):
        nbl = nant*(nant+1)//2
        self.nbl = nbl
        self.nant = nant
        self.nt = nt
        self.npol = npol
        self.nc = nc
        self.vis_fscrunch = vis_fscrunch
        self.vis_tscrunch = vis_tscrunch
        self.rescale_update_blocks = rescale_update_blocks
        self.dtype = get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, rdtype, cdtype)
        self.output = np.zeros(nbeam, dtype=self.dtype)
        self.rescale_stats = np.zeros((nbeam, nc, self.nbl, npol, 2), dtype=rdtype)
        self.rescale_scales = np.zeros((nbeam, nc, self.nbl, npol, 2), dtype=rdtype)
        self.count = np.zeros(NFPGA, dtype=np.int32)

        assert self.output[0]['cas'].shape == self.output[0]['ics'].shape, f"do_accumulate assumes cas and ICS work on same shape. CAS shape={self.output[0]['cas'].shape} ICS shape={self.output[0]['ics'].shape}"

        if exclude_ants is None:
            exclude_ants = []

        self.exclude_ants = set(map(int, exclude_ants))
        self.antenna_mask = np.array([False if (iant+1) in exclude_ants else True for iant in range(nant)])
        log.info('There are %s valid antennas in mask=%s. Excluding=%s',
                 sum(self.antenna_mask==True), self.antenna_mask, self.exclude_ants)
        assert len(self.antenna_mask) == nant
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
            packets = [(0, self.dummy_packet) for i in range(NFPGA)]

            self.accumulate_packets(packets)
            self.iblk = 0

        self.reset()
        self.reset_scales()

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

        valid = np.array([pkt[1] is not None for pkt in packets], dtype=bool)
        data = List()
        [data.append(self.dummy_packet if pkt[1] is None else pkt[1]) for pkt in packets]
        self.reset()
        return self.accumulate_all(data, valid)


    def accumulate_all(self, beam_data, valid):
        '''
        Runs multi-threaded accumulation over all fpgas/coarse channels / beams / times /  baselnes
        :param: beam_data is numba List with the expected data
        '''
        
        accumulate_all(self.output,
                       self.rescale_scales,
                       self.rescale_stats,
                       self.count,
                       self.nant,
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


    def accumulate_beam(self, ibeam, ichan, beam_data):
        do_accumulate(self.output, self.rescale_scales, self.rescale_stats, self.count, self.nant, ibeam, ichan, beam_data, self.vis_fscrunch, self.vis_tscrunch)

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
