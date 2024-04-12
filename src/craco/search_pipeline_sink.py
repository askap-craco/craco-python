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
from craco.timer import Timer
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
    def __init__(self, info, iblk):
        self.info = info
        self.flag_ants = []
        self.iblk = iblk
        assert iblk >= 0
        self._baselines = None

    def set_flagants(self, flag_ants):
        '''
        Set flag antennas as a list of 1-based integers
        '''
        self.flag_ants = list(flag_ants)

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
        if self._baselines is not None:
            return self._baselines
        # oooh, gee, this is going to be a headache if we get it wrong.
        # how does flagants work?
        # bleach. Gross. This requires much thinking
        # for now we just do something brittle and see where it breaks

        # calculate the block when this VisInfo finishes
        nblk = self.info.values.update_uv_blocks
        assert nblk >= 0,'Invalid update uv blocks'
        start_fid = self.info.fid_of_block(self.iblk)  # Frame ID of beginning
        # fid_mid is the frame ID of hte middle of the block starting at the beginning of iblk
        # and finishing at the beginning of iblk+nblk
        end_fid = self.info.fid_of_block(self.iblk+nblk)
        fid_mid = start_fid + (end_fid - start_fid) // 2

        # nblk == 0 is disabled. fid_mid will be the beginning of the block. You have been warned
        mjd_mid = self.info.fid_to_mjd(fid_mid)
        log.info('Returning baselines for iblk=%s start_fid=%s fid_mid=%s mjd_mid=%s tstart=%s',
                 self.iblk, start_fid, fid_mid, mjd_mid, self.tstart)

        
        self._baselines = self.info.baselines_at_time(mjd_mid)
        return self._baselines

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
    def freq_config(self):
        '''
        Returns the frequency configuration object
        '''
        return self.info.vis_freq_config

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
        Returns the start time for this visinfo
        Not necessarily the start time for the whole observation
        '''
        fid = self.info.fid_of_block(self.iblk)
        mjd = self.info.fid_to_mjd(fid)
        return mjd


    @property
    def tsamp(self):
        '''
        Return sample interval in .... seconds?
        '''
        return self.info.vis_inttime


class SearchPipelineSink:
    def __init__(self, info):
        self.info = info
        self.iblk = 0
        self.adapter = VisInfoAdapter(self.info, self.iblk)
        self.last_write_timer = Timer()

        devid = info.xrt_device_id
        self.pipeline = None
        self._next_plan_data = None
        if devid is not None:
            log.info('Beam %s Loading device %s with %s', info.beamid, devid, info.values.xclbin)
            try:
                self.pipeline = PipelineWrapper(self.adapter, info.values, devid)
                nf = len(info.vis_channel_frequencies)
                nt = self.pipeline.plan.nt
                nbl = self.adapter.nbl
                shape = (nbl, nf, nt)
                self.pipeline_data = np.ma.masked_array(np.zeros(shape, dtype=np.complex64), mask=np.zeros(shape, dtype=bool))

                nfcas = info.nchan
                cas_shape = (nfcas, nt)
                ##self.cas_data = np.zeros(cas_shape, dtype=np.float32)
                #self.ics_data = self.cas_data.copy()
                
                self.t = 0
            except RuntimeError: # usually XRT error
                log.exception(f'Failed to make pipeline for devid={devid}. Ignoring this pipeline')
                self.pipeline = None

    def set_next_plan(self, next_plan_data):
        '''
        Update the value of the next plan. It might not necssarily be used immediately, but
        we'll have it in hand just in case. The plan is a craco-plan that was made in a separate process
        '''
        log.info('Got next plan %s', next_plan_data)
        self._next_plan_data = next_plan_data

    @property
    def ready_for_next_plan(self):
        '''
        Returns True if we're ready to accept the next plan
        '''
        return self._next_plan_data is None

    def write(self, vis_block):
        '''
        vis_data has len(nrx) and shape inner shape
        vishape = (nbl, vis_nc, vis_nt, 2) if np.int16 or
        or 
        vishape = (nbl, vis_nc, vis_nt) if np.complex64
        '''
        if self.pipeline is None:
            return

        t = Timer()
        vis_data = vis_block.data
        
        # TODO: convert input beam data to whatever search_pipeline wants
        # which is an array of [nbl, nf, nt]
        # Don't forget, the data itself might need to be blocked into nt=256 which is what the pipeline wants
        # copy data into local buffer
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert vis_data.dtype == np.complex64, f'I think we can only handle complex data in this function. Vis data type was {vis_data.dtype} {vis_data.shape}'
        output_nt = self.pipeline_data.shape[2]
        
        assert output_nt % vis_nt == 0, f'Output must be a multiple of input NT. output={output_nt} vis={vis_nt} vis_data.shape'
        assert vis_nc*nrx == self.pipeline_data.shape[1], f'Output NC should be {self.pipeline_data.shape[1]} but got {vis_nc*nrx} {vis_data.shape}'
        assert self.pipeline_data.shape[0] == nbl, f'Expected different nbl {self.pipeline_data.shape} != {nbl} {vis_data.shape}'

        info = self.info
        blflags = vis_block.baseline_flags[:,np.newaxis,np.newaxis]

        tstart = self.t
        tend = tstart + vis_nt

        # loop through each card
        for irx in range(nrx):
            fstart = irx*vis_nc
            fend = fstart + vis_nc
            chanmask = abs(vis_data[irx, ...]) == 0

            self.pipeline_data[:,fstart:fend, tstart:tend] = vis_data[irx, ...]
            self.pipeline_data.mask[:,fstart:fend, tstart:tend] = chanmask | blflags


            cas_fslice = slice(fstart*6,fend*6)
            # # need to fix shapes - but viveks' new prepare pipeline does it anyway
            #self.cas_data[cas_fslice, tstart:tend] = vis_block.cas[irx,...].T 
            #self.ics_data[cas_fslice, tstart:tend] = vis_block.ics[irx,...].T


        self.t += vis_nt

        t.tick('Copy')

        # Update UVWs if necessary
        # Don't do it on block 0 as we've already made one
        # Don't do it if disabled with values.update_uv_blocks = 0
        update_uv_blocks = self.info.values.update_uv_blocks
        update_now = update_uv_blocks > 0 and self.iblk % update_uv_blocks == 0 and self.iblk != 0
        if update_now:
            #self.adapter = VisInfoAdapter(self.info, self.iblk)
            #self.pipeline.update_plan(self.adapter)
            pd = self._next_plan_data
            assert pd['iblk'] == self.iblk, f'Got plan to apply at wrong time. my iblk={self.iblk} plan iblk={pd["iblk"]}'
            self.pipeline.update_plan_from_plan(pd['plan'])
            self._next_plan_data = None # set it to None ready for the next plan
            
        t.tick('Update plan')

        if self.t == output_nt:
            try:
                self.pipeline.write(self.pipeline_data)
                t.tick('Pipeline write')
                self.t = 0
            except RuntimeError: # usuall XRT error
                log.exception('Error sending data to pipeline. Disabling this pipeline')
                self.pipeline.close()
                self.pipeline = None

        self.iblk += 1
        self.last_write_timer = t
            
        
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
