#!/usr/bin/env python
"""
Utilities to average cardcap data from a card

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging

from numba import jit,njit,prange
import numba

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

@njit(fastmath=True,parallel=False, locals={'v0':numba.float32, 'v1':numba.float32, 'vsqr':numba.float32,'va':numba.float32})
def do_accumulate(output, rescale_scales, rescale_stats, nant, ibeam, ichan, beam_data, vis_fscrunch=1, vis_tscrunch=1):
    '''
    Computes ICS, CAS and averaged visibilities given a block of nt integration packets from a single FPGA
    
    makeing it parallel makes it worse
    :output: Rescaled and averaged output. 1 per beam.
    :rescale_scales: (nbeam, nbl, npol, 2) float 32 scales to apply to vis amplitudes before adding in ICS/CAS. [0] is subtracted and [1] is multiplied. 
    :rescale_stats: (nbeam, nbl, npol, 2) flaot32 statistics of the visibility amplitues. [0] is the sum and [2] is the sum^2
    :nant: number of antnenas. Should tally with number of baselines
    :ibeam: Beam number to udpate
    :ichan: Channel number to update
    :vis_fscrunch: Visibility fscrunch factor
    :vis_tscrunch: Visibility tscrunch factor
    
    '''

    ics = output[ibeam]['ics']
    cas = output[ibeam]['cas']
    vis = output[ibeam]['vis']
    nsamp = len(beam_data)
    (nsamp2, nbl, npol, _) = beam_data[0]['data'].shape
    for samp in range(nsamp):
        bd = beam_data[samp]['data']
        for samp2 in range(nsamp2):
            t = samp2 + nsamp2*samp
            ochan = ichan // vis_fscrunch
            otime = t // vis_tscrunch
            a1 = 0
            a2 = 0
            # looping over baseline and calculating antenna numbers is about 15% faster than
            # 2 loops over antennas
            for ibl in range(nbl):
                for pol in range(npol):
                    v = bd[samp2, ibl, pol, :]
                    v0 = np.float32(v[0])
                    v1 = np.float32(v[1])
                    vsqr = v0*v0 + v1*v1
                    va = np.sqrt(vsqr)
                    va = vsqr
                    rescale_stats[ibeam, ibl, pol, 0] += va # Update amplitude
                    rescale_stats[ibeam, ibl, pol, 1] += vsqr # Add amplitude**2 for stdev
                    mean = rescale_scales[ibeam, ibl, pol, 0]
                    scale = rescale_scales[ibeam, ibl, pol, 1]
                    va_scaled = (va - mean)*scale
                        
                    if a1 == a2:
                        # Could just add a scaled version of v0 here, but it makes  little diffeence
                        # given there are so few autos
                        ics[t,ichan] += va_scaled
                    else:
                        cas[t,ichan] += va_scaled
                        
                    vis[ibl, ochan, otime] += complex(v0,v1)

                    
                a2 += 1
                if a2 == nant -1:
                    a1 += 1
                    a2 = a1
                    
@njit(fastmath=True, parallel=True)
def accumulate_all(output, rescale_scales, rescale_stats, nant, beam_data, vis_fscrunch=1, vis_tscrunch=1):
    nfpga= len(beam_data)
    npkt = len(beam_data[0])
    nbeam, nbl, npol, _ = rescale_scales.shape
    nc_per_fpga = 4
    npkt_per_accum = npkt // (nbeam * nc_per_fpga)
    for beam in prange(nbeam):
        for fpga in range(6):
            for chan in range(4):
                data = beam_data[fpga]
                if beam < 32:
                    didx = beam + 32*chan
                else:
                    b = beam - 32
                    didx = 32*4 + b + 4*chan

                #assert didx < 4*36

                ichan = fpga + 6*chan # TODO: CHECK
                startidx = didx*npkt_per_accum
                endidx = startidx + npkt_per_accum
                #print(beam, fpga, chan, didx, len(beam_data), data.shape, len(data),  npkt_per_accum, startidx,endidx)
                #assert endidx <= len(data)
                bd = data[startidx:endidx]
                do_accumulate(output, rescale_scales, rescale_stats, nant, beam, ichan, bd, vis_fscrunch, vis_tscrunch)


                
class Averager:
    def __init__(self, nbeam, nant, nc, nt, npol, rdtype=np.float32, cdtype=np.complex64, vis_fscrunch=6, vis_tscrunch=2):
        nbl = nant*(nant+1)//2
        self.nbl = nbl
        self.nant = nant
        self.nt = nt
        self.npol = npol
        self.vis_fscrunch = vis_fscrunch
        self.vis_tscrunch = vis_tscrunch
        assert nt % vis_tscrunch == 0, 'Tscrunch should divide into nt'
        assert nc % vis_fscrunch == 0, 'Fscrunch should divide into nc'
        vis_nt = nt // vis_tscrunch
        vis_nc = nc // vis_fscrunch
        self.dtype = np.dtype([('ics', rdtype, (nt, nc)),
                               ('cas', rdtype, (nt, nc)),
                               ('vis', cdtype, (nbl, vis_nc, vis_nt))])
        self.output = np.zeros(nbeam, dtype=self.dtype)
        self.rescale_stats = np.zeros((nbeam, self.nbl, npol, 2), dtype=rdtype)
        self.rescale_scales = np.zeros((nbeam, self.nbl, npol,  2), dtype=rdtype)

    def reset():
        self.output[:] = 0
        self.rescale_stats[:] = 0

    @njit(fasttmath=True, parallel=True)
    def accumulate_beam(self, ibeam, ichan, beam_data):
        do_accumulate(self.output, self.rescale_scales, self.rescale_stats, self.nant, ibeam, ichan, beam_data)


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
