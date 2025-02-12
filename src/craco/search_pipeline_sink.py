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
from craco.candidate_writer import CandidateWriter
from craco.mpi_appinfo import MpiPipelineInfo
from craco.mpi_obsinfo import MpiObsInfo


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

    def __str__(self):
        s = f'VisInfo beam={self.beamid} iblk={self.iblk} tstart={self.tstart} maxuv={self.get_max_uv()} target={self.target_name} {self.target_skycoord} freq={self.freq_config} tsamp={self.tsamp}'

        return s

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

        # calculate the block when this VisInfo finishe
        nblk = self.info.values.update_uv_blocks
        assert nblk >= 0,'Invalid update uv blocks'
        start_fid = self.info.fid_of_search_block(self.iblk)  # Frame ID of beginning
        # fid_mid is the frame ID of hte middle of the block starting at the beginning of iblk
        # and finishing at the beginning of iblk+nblk
        end_fid = self.info.fid_of_search_block(self.iblk+nblk)
        mid_fid = start_fid + np.uint64((end_fid - start_fid) // 2)

        # nblk == 0 is disabled. fid_mid will be the beginning of the block. You have been warned - in a comment that no-one will read
        mjd_mid = self.info.fid_to_mjd(mid_fid)
        mjd_start = self.info.fid_to_mjd(start_fid)
        mjd_end = self.info.fid_to_mjd(end_fid)
        tstart = self.info.tstart
        middif = (mjd_mid - tstart).to('s')
        expect_mjd_mid = tstart + self.tsamp*256*(self.iblk + 0.5)

        log.info('Returning baselines for iblk=%s FID: %s/%s/%s MJD %s/%s/%s expect mid MJD: %s mid-tstart=%s (sec)', 
                 self.iblk,
                 start_fid, mid_fid, end_fid, 
                 mjd_start.iso, mjd_mid.iso, mjd_end.iso, 
                 expect_mjd_mid.iso, middif)
        
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
    def __init__(self, info:MpiObsInfo, plan):
        '''
        Includes info and pre-made initial plan. 
        '''
        self.info = info
        self.iblk = 0
        self.adapter = VisInfoAdapter(self.info, self.iblk)
        self.last_write_timer = Timer()

        devid = info.xrt_device_id
        self.pipeline = None
        self._next_plan_data = None
        log.info('SearchPipelineSink devid=%s beamid=%s search_beams=%s', devid, self.info.beamid, self.info.values.search_beams)
        if devid is not None and self.info.beamid in self.info.values.search_beams:
            log.info('Beam %s Loading device %s with %s', info.beamid, devid, info.values.xclbin)
            try:
                self.pipeline = PipelineWrapper(self.adapter, info.values, devid, parallel_mode=True, plan=plan)
                nf = len(info.vis_channel_frequencies)
                nt = self.pipeline.plan.nt
                nbl = self.adapter.nbl
                shape = (nbl, nf, nt)
                self.pipeline_data = np.ma.masked_array(np.zeros(shape, dtype=np.complex64), mask=np.zeros(shape, dtype=bool))
                nfcas = info.nchan
                #cas_shape = (nfcas, nt)
                ##self.cas_data = np.zeros(cas_shape, dtype=np.float32)
                #self.ics_data = self.cas_data.copy()
                
                self.t = 0
            except RuntimeError: # usually XRT error
                log.exception(f'RuntimeError to make pipeline for devid={devid}. Ignoring this pipeline')
                self.pipeline = None
            except MemoryError: # useually pyxrt.syncbo fails due to dead card
                log.exception(f'MemoryError to make pipeline for devid={devid}. Ignoring this pipeline')
                self.pipeline = None
        else:
            log.info('SearchPielineSink not searching beamid=%s, device or search beam not specified', self.info.beamid)
                

    def set_next_plan(self, next_plan_data):
        '''
        Update the value of the next plan. It might not necssarily be used immediately, but
        we'll have it in hand just in case. The plan is a craco-plan that was made in a separate process
        '''
        next_iblk = next_plan_data['iblk']
        self._next_plan_data = next_plan_data
        log.info('Got next plan for iblk %d. Current iblk =%d. Advance=%d', next_iblk, self.iblk, next_iblk - self.iblk)


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

        assert False, 'Do not use. Were now accumulating blocks in another MPI process'
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
        self.write_pipeline_data(pipeline_data)

    def write_pipeline_data(self, pipeline_data, candidate_buffer):
        '''
        Writes already accumulated pipeline data
        Should be a masked array of dtype=compelx64 and shape
        (nbl, nf, nt=256)

        Also updates plan if necessary
        '''
        t = Timer()
        save_pipeline_data = True
        if save_pipeline_data:
            pth = os.path.join(self.info.beam_dir, f'pipeline_data_{self.iblk}.npy')
            np.save(pth, pipeline_data)
            t.tick('Save pipeline data', args={'iblk':self.iblk, 'path':pth})
            log.info('Saved pipeline data to %s iblk=%s', pth, self.iblk)

        out_cands = None
        if self.pipeline is None:
            return out_cands
        

        # Update UVWs if necessary
        # Don't do it on block 0 as we've already made one
        # Don't do it if disabled with values.update_uv_blocks = 0
        update_uv_blocks = self.info.values.update_uv_blocks
        update_now = update_uv_blocks > 0 and self.iblk % update_uv_blocks == 0 and self.iblk != 0
        if update_now:
            pd = self._next_plan_data # comes from another MPI rank
            if pd is None:
                raise ValueError(f'Need to update but no plan available. iblk={self.iblk}')
            assert pd['iblk'] == self.iblk, f'Got plan to apply at wrong time. my iblk={self.iblk} plan iblk={pd["iblk"]}'
            self.pipeline.update_plan_from_plan(pd['plan'])
            self._next_plan_data = None # set it to None ready for the next plan            
            t.tick('Update plan')        

        try:
            vis = pipeline_data['vis']
            bl_weights = pipeline_data['bl_weights']
            tf_weights = pipeline_data['tf_weights']
            summarise_input = True

            # this can take a long time, especially on a slow node - like 1000ms
            if summarise_input:
                log.info('Pipeline data iblk=%d abs(vis).mean=%0.1e bl_weights=%s/%s tf_weights=%s/%s',
                         self.iblk,
                         abs(vis).mean(),
                         bl_weights.sum(), bl_weights.size,
                         tf_weights.sum(), tf_weights.size)
                t.tick('Summarise input')

            # Originally the dtype in the VisblockAccumulatorStruct was bool, but we
            # had to numbafy it. Numba doesn't like bools, so I converted to uint8.
            # I'm not sure that the search pipeline (inparticualr fast preprorcessor)
            # likes np.uint8, so I send through a bool view here. 
            # It might work without it but I can't face any more dumb bugs right now.
            out_cands = self.pipeline.write(vis, 
                                            bl_weights=bl_weights.view(dtype=bool), 
                                              input_tf_weights=tf_weights.view(dtype=bool), 
                                              candout_buffer=candidate_buffer) 
            t.tick('Pipeline write')
            self.t = 0
        except RuntimeError: # usuall XRT error
            log.exception('Error sending data to pipeline. Disabling this pipeline')
            self.pipeline.close()
            self.pipeline = None
        except:
            dumpfile = os.path.abspath(f'pipeline_sink_dump.npz')
            log.exception('Some error. saving data to %s', dumpfile)
            np.save(dumpfile, pipeline_data)
            sz = os.path.getsize(dumpfile)
            log.info('Saved %d bytes to %s', sz, dumpfile)
            raise


        self.iblk += 1
        self.last_write_timer = t

        return out_cands            
        
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
