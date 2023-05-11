#!/usr/bin/env python
"""
Data sink for the MPIPIPeline to send to the saerch pipeline

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craft.craco import ant2bl,get_max_uv
from craft.craco_plan import PipelinePlan
from craco.search_pipeline import PipelineWrapper
from astropy import units as u
import pyxrt
import scipy

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class VisInfoAdapter:
    '''
    Adapts a pipeline info into the thing that pipeline plan needs to make a plan
    Modeled on craft.uvfits
    '''
    def __init__(self, info):
        self.info = info
        self.flag_ants = []

    def set_flagants(self, flag_ants):
        '''
        Set flag antennas as a list of 1-based integers
        '''
        self.flag_ants = flag_ants

    def get_max_uv(self):
        ''' Return umax, vmax'''
        fmax = self.channel_frequencies.max()
        baselines = self.baselines
        maxuv = get_max_uv(baselines, fmax)
        return maxuv

    @property
    def baselines(self):
        '''
        Returns dictionary by blid of baselines, containing UU,VV,WW in seconds
        Like a uv fits file
        '''
        # oooh, gee, this is going to be a headache if we get it wrong.
        # how does flagants work?
        # bleach. Gross. This requires much thinking
        # for now we just do something brittle and see where it breaks
        toffset = 10*u.minute # TODO: How far in the future should we compute the UVWs?
        tbaseline = self.tstart + toffset
        # UVW is a np array shape [nant, 3]
        uvw = self.info.uvw_at_time(tbaseline)
        bluvws = {}

        for blinfo in self.info.baseline_iter():
            bluvw = uvw[blinfo.ia1, :] - uvw[blinfo.ia2, :]
            assert np.all(bluvw != 0), f'UVWs were zero for {blinfo}'
            d = {'UU': bluvw[0], 'VV': bluvw[1], 'WW':bluvw[2]}
            bluvws[float(blinfo.blid)] = d

        return bluvws

    @property
    def nbl(self):
        return len(self.baselines)

    @property
    def channel_frequencies(self):
        '''
        Return np array of channel frequencies in Hz.
        '''
        return self.info.vis_channel_frequencies*1e6

    @property
    def target_name(self):
        '''
        String target name
        '''
        s = f'{self.info.target}_beam{self.info.beamid:02d}'
        return s

    @property
    def target_skycoord(self):
        '''
        Return target as SkyCoord
        '''
        return self.info.skycoord

    @property
    def beamid(self):
        return self.info.beamid

    @property
    def tstart(self):
        '''
        return start time as astropy time
        '''
        return self.info.tstart

    @property
    def tsamp(self):
        '''
        Return sample interval in .... seconds?
        '''
        return self.info.vis_inttime

class SearchPipelineSink:
    def __init__(self, info):
        self.info = info
        self.adapter = VisInfoAdapter(self.info)

        devid = info.xrt_device_id
        self.pipeline = None
        if devid is not None:
            log.info('Beam %s Loading device %s with %s', info.beamid, devid, info.values.xclbin)
            try:
                self.pipeline = PipelineWrapper(self.adapter, info.values, devid)
                nf = len(info.vis_channel_frequencies)
                nt = self.pipeline.plan.nt
                nbl = self.adapter.nbl
                self.pipeline_data = np.zeros((nbl, nf, nt), dtype=np.complex64)
                self.t = 0
            except:
                log.exception(f'Failed to make pipeline for devid={devid}')
                raise
            

    def write(self, vis_data):
        '''
        vis_data has len(nrx) and shape inner shape
        vishape = (nbl, vis_nc, vis_nt, 2) if np.int16 or
        or 
        vishape = (nbl, vis_nc, vis_nt) if np.complex64
        '''
        if self.pipeline is not None:
            # TODO: convert input beam data to whatever search_pipeline wants
            # which is an array of [nbl, nf, nt]
            # Don't forget, the data itself might need to be blocked into nt=256 which is what the pipeline wants
            # copy data into local buffer
            nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
            assert vis_data.dtype == np.complex64, 'I think we can only handle complex data in this function'
            output_nt = self.pipeline_data.shape[2]

            assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
            assert vis_nc*nrx == self.pipeline_data.shape[1], f'Output NC should be {self.pipeline_data.shape[1]} but got {vis_nc*nrx} {vis_data.shape}'
            assert self.pipeline_data.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'

            for irx in range(nrx):
                fstart = irx*vis_nc
                fend = fstart + vis_nc
                tstart = self.t
                tend = tstart + vis_nt
                self.pipeline_data[:,fstart:fend, tstart:tend] = vis_data[irx, ...]

            self.t += vis_nt

            if self.t == output_nt:
                self.pipeline.write(self.pipeline_data)
                self.t = 0
            
        
    def close(self):
        '''
        Tidy up, probably dont need to do much
        '''
        if self.pipeline is not None:
            self.pipeline.close()
        

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
