#!/usr/bin/env python
import numpy as np
from pylab import *
import os
import pyxrt
from .pyxrtutil import *
import time
import pickle
import copy
from astropy.time import Time
from craft.cmdline import strrange

import craft.craco_plan

from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import load_plan

from craft.craco import printstats
from craft import sigproc

from craft import uvfits
from craft import craco
from craco import calibration, uvfits_meta
from craco.timer import Timer
from craco.vis_subtractor import VisSubtractor
from craco.vis_flagger import VisFlagger
from craco.preprocess import FastPreprocess, TAB_handler
from craco.candidate_writer import CandidateWriter
from craco import write_psf as PSF

from Visibility_injector.inject_in_fake_data import FakeVisibility

from collections import OrderedDict
#from trace_event_handler import TraceEventHandler
from astropy import units

import logging
import warnings

DM_CONSTANT = 4.15 # milliseconds and GHz- Shri Kulkarni will kill me

#handler = TraceEventHandler()
#logging.basicConfig(handlers=[logging.StreamHandler(None), handler], level=logging.INFO)
log = logging.getLogger(__name__)


#from craco_pybind11 import boxcar, krnl, fdmt_tunable

'''
Most hard-coded numebrs are updated
'''

# ./test_pipeline.py -R -r -T 1.0 -u frb_d0_t0_a1_sninf_lm00.fits
# search for
# /data/craco/den15c/old_pipeline/realtime_pipeline/imag_fixed_noprelink/golden_candidate has candidates

candidate_dtype = [('snr', np.int16),
                   ('loc_2dfft', np.uint16),
                   ('boxc_width', np.uint8),
                   ('time', np.uint8),
                   ('dm', np.uint16)]

NBLK = 5
NCU = 4
NTIME_PARALLEL = (NCU*2)

#NDM_MAX = krnl.MAX_NDM
NDM_MAX = 1024
HBM_SIZE = int(256*1024*1024)
NBINARY_POINT_THRESHOLD = 6
NBINARY_POINT_FDMTIN    = 5
GRID_LUT_MAX = 180214 # TODO: work out how to calculate this number


def make_fft_config(nplanes:int,
                    stage1_scale:int = 0,
                    stage2_scale:int = 7,
                    bypass_stage1:bool = False,
                    bypass_transpose:bool = False,
                    bypass_stage2:bool = False) -> int:
    '''
    Generates the FFT config word to execute the FFT.

    The layout of this word I got from unzipping fft2d_v2020.2_rtl_d09062021.xo
    And lookin at TEST_SystyolicFFT2D.vhd and fft2d.v, and the result is
    bit 0 - if 1, bypass stage 1 FFT
    bit 1 - if 1, bypass tranpsose stage
    bit 2 - if 1, bypass stage 2 FFT
    bit 3-5 - shifting RIGHT apply at input of the first stage FFT
    bit 6-8 - shifting LEFT to apply at input of the 2nd stage FFT
    bits 16-31 - number of planes to process

    Note: the shift values operate in OPPOSITE SENSES for the 1st and 2nd stage. 
    In the first stage, it shifts RIGHT (i.e. divides by 2^N). In the 2nd stage
    it shifts LEFT (I.e. multiplies by 2^N). 
    With both scale factors equal to zero (i.e. no scaling) the output equals 
    the sum of the inputs times 2^{-10} which implies there's 5 bits of round/truncation happening
    in each FFT stage.


    :nplanes: Number of FFTs to do in this invocation
    :stage1_scale: Shift input LEFT by stage1_scale bits before 1st stage
    :stage2_scale: Shift intermediate data RIGHT by stage2_scale bits before 2nd stage
    :bypass_stage1: Set to True to bypass stage 1 FFT
    :bypass_stage2: Set to True to bypass stage 2 FFT
    :bypass_transpose: Set to Try to bypass transpose

    :returns: FFT config word
    '''
    assert 0 < nplanes < 1<<16
    assert 0 <= stage1_scale < 1<<3
    assert 0 <= stage2_scale << 1<<3

    by_s1 = int(bypass_stage1)
    by_t = int(bypass_transpose)
    by_s2 = int(bypass_stage2)

    word = (nplanes & 0xffff) << 16 | \
            (stage2_scale & 0b111) << 6 | \
            (stage1_scale & 0b111) << 3 | \
            by_s2 << 2 | \
            by_t << 1 | \
            by_s1

    return word

def count_candidates(c):
    '''
    Count the number of candidates in a block of candiates by finding the first candidate with snr==0
    returns the index of that candidate. If there are no candidates in the block then returns len(c)
    '''
    try:
        ncand, lastcand = next((i,c) for i,c in enumerate(c) if c['snr']  == 0)
    except StopIteration:
        ncand = len(c) 

    return ncand


def calc_fft_scale(fft_shift1, fft_shift2):
    '''
    Returns the scaling in accross the FFT2D kernel
    fft_shift1 divides by 2**fft_shift1
    fft_shift2 multiplies by 2**fft_shift2
    
    The whole FFT also divides by 2**10

    So the output is scale = 2**(fft_shift2 - fft_shift1 - 10)

    e.g. for the DC-bin:
    i.e. fft_output[0] = sum(fft_input)*calc_fft_scaling(fft_shift1, fft_shift2)
    '''
    assert 0 <= fft_shift1 <= 7
    assert 0 <= fft_shift2 <= 7
    scale = 2**(fft_shift2 - fft_shift1 - 10)
    
    return scale

def merge_candidates_width_time(cands):
    '''
    Merges candidates that have overlappign width and times for identical DMs and locations
    Assumes data in hardware ordering (whatever that is)

    THIS IT NOT VERY EFFICIENT! But it will work for now

    It is very complicated

    :returns: New candidate list
    '''
    cout = []
    if len(cands) == 0:
        return cands
    
    cands = np.sort(cands, order=['dm', 'loc_2dfft', 'time', 'boxc_width', 'snr'])
    curr_cand = [cands[0]]
    for icand, cand in enumerate(cands[1:]):
        # if it's the same place and DM
        if cand['dm'] == curr_cand[0]['dm'] and cand['loc_2dfft'] == curr_cand[0]['loc_2dfft']:
            # if the current candidate is in time window of the previous one, then group it
            # calcualate start and end times for hte current and first andidates
            # times are inclusive 
            curr_cand_start_time = cand['time'] - cand['boxc_width'] 
            curr_cand_end_time = cand['time'] 
            first_cand_start_time = curr_cand[0]['time'] - cand['boxc_width']
            first_cand_end_time = curr_cand[0]['time']
            if first_cand_start_time <= curr_cand_start_time <= first_cand_end_time:
                curr_cand.append(cand)
            else:
                best_cand =  max(curr_cand, key=lambda c:c['snr'])
                yield best_cand
                curr_cand = [cand]
        else: # change of pixelor DM
            best_cand =  max(curr_cand, key=lambda c:c['snr'])
            yield best_cand
            curr_cand = [cand]

    if len(curr_cand) > 0:
        best_cand =  max(curr_cand, key=lambda c:c['snr'])
        yield best_cand
        

def get_mode():
    mode = os.environ.get('XCL_EMULATION_MODE', 'hw')
    return mode
    
class DdgridCu(Kernel):
    def __init__(self, device, xbin):
        super().__init__(device, xbin, 'krnl_ddgrid_reader_4cu')
  
        
class FfftCu(Kernel):
    def __init__(self, device, xbin, icu):
        super().__init__(device, xbin, 'fft2d',icu=icu)
 
class GridCu(Kernel):
    def __init__(self, device, xbin, icu):
        super().__init__(device, xbin, f'krnl_grid_4cu', icu=icu)

class BoxcarCu(Kernel):
    def __init__(self, device, xbin):
        super().__init__(device, xbin, f'krnl_boxc_4cu')

class FdmtCu(Kernel):
    def __init__(self, device, xbin):
        super().__init__(device, xbin, 'fdmt_tunable_c32')

def instructions2grid_lut(instructions, max_nsmp_uv):
    data = np.array([[i.target_slot, i.uvidx, i.shift_flag, i.uvpix[0], i.uvpix[1]] for i in instructions], dtype=np.int32)
    
    nuvout = len(data[:,0]) # This is the number of output UV, there are zeros in between

    output_index = data[:,0]
    input_index  = data[:,1]
    send_marker  = data[:,2][1::2]

    nuvin = np.sort(input_index)[-1]   # This is the real number of input UV

    max_nparallel_uv = max_nsmp_uv//2
    output_index_hw = np.pad(output_index, (0, max_nsmp_uv-nuvout), 'constant')
    input_index_hw  = np.pad(input_index,  (0, max_nsmp_uv-nuvout), 'constant')
    send_marker_hw  = np.pad(send_marker,  (0, max_nparallel_uv-int(nuvout//2)), 'constant')
    
    return nuvin, nuvout, input_index_hw, output_index_hw, send_marker_hw

def instructions2pad_lut(instructions, npix):
    location = np.zeros(npix*npix, dtype=int)

    data = np.array(instructions, dtype=np.int32)
    upix = data[:,0]
    vpix = data[:,1]

    location_index = vpix*npix+upix
    #location_index = ((npix_half+vpix)%npix)*npix + (npix_half+upix)%npix
    #((FFT_SIZE/2+vpix)%FFT_SIZE)*FFT_SIZE +
    #(FFT_SIZE/2+upix)%FFT_SIZE;

    location_value = data[:,2]+1

    location[location_index] = location_value

    return location
    
def get_grid_lut_from_plan(plan):

    # Hack - get fdmt plan to set values
    fplan = plan.fdmt_plan
    upper_instructions = plan.upper_instructions
    lower_instructions = plan.lower_instructions

    # careful here as craco_plan define nuvmax to be multiple times 8, but we need 2 extra space to pack
    max_nsmp_uv = plan.nuvmax-2
    nuv, nuvout, input_index, output_index, send_marker           = instructions2grid_lut(upper_instructions, max_nsmp_uv)
    h_nuv, h_nuvout, h_input_index, h_output_index, h_send_marker = instructions2grid_lut(lower_instructions, max_nsmp_uv)

    if nuv != h_nuv:
        warnings.warn(f'nuv={nuv} shouldS equal h_nuv={h_nuv}. Or ... should it? THis happens becasuse the lower matrix doesnt include the diagonal and the last UV index used is on the diagona.. Well take the maximum here but perhaps we should have thouught a bit harder. See CRACO-125 for more info')
        nuv = max(nuv, h_nuv)

    nuv_round = nuv+(8-nuv%8)       # Round to 8
    location   = instructions2pad_lut(plan.upper_idxs, plan.npix)
    h_location = instructions2pad_lut(plan.lower_idxs, plan.npix)
    
    shift_marker   = np.array(plan.upper_shifts, dtype=np.uint16)
    h_shift_marker = np.array(plan.lower_shifts, dtype=np.uint16)
    
    lut = np.concatenate((output_index, input_index, send_marker, location, shift_marker, h_output_index, h_input_index, h_send_marker, h_location, h_shift_marker)).astype(np.uint16)
    
    return nuv_round//2, nuvout//2, h_nuvout//2, lut


class Pipeline:
    def __init__(self, device, xbin, plan, alloc_device_only_buffers=False):
        '''
        Make the search pipeline bound to the hardware
        
        :device: XRT device
        :xbin: XCLBIN object
        :plan:PIpeline plan
        :alloc_device_only_buffers: Set to True if you want to be able to manipulate buffers that we normally only use fo rdevice only
        '''
        
        self.device = device
        self.xbin = xbin
        self.plan = plan
        self.alloc_device_only_buffers = alloc_device_only_buffers
        self.device_only_buffer_flag = 'normal' if alloc_device_only_buffers else 'device_only'

        self.grid_reader = DdgridCu(device, xbin)
        self.grids = [GridCu(device, xbin, i) for i in range(4)]
        self.ffts = [FfftCu(device, xbin, i) for i in range(4)]
        self.boxcarcu = BoxcarCu(device, xbin)
        self.fdmtcu = FdmtCu(device, xbin)
        self.all_kernels = [self.grid_reader, self.fdmtcu, self.boxcarcu]
        self.all_kernels.extend(self.grids)
        self.all_kernels.extend(self.ffts)


                
        
        log.info('Allocating FDMT Input')

        # Need ??? BM, have 5*256 MB in link file, but it is not device only, can only alloc 256 MB?
        #self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.ncin, self.plan.nt, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()
        #self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.nt, self.plan.ncin, self.plan.nuvwide, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()
        inbuf_shape = (self.plan.nuvrest_max, self.plan.nt, self.plan.ncin, self.plan.nuvwide, 2)
        self.inbuf = Buffer(inbuf_shape, np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()

        log.info(f'FDMT input buffer size {np.prod(inbuf_shape)*2/1024/1024} MB')
        
        # FDMT histin, histhout should be same buffer
        assert self.fdmtcu.group_id(2) == self.fdmtcu.group_id(3), 'FDMT histin and histout should be the same'
        
        log.info('Allocating FDMT history')
        # Need ??? MB, we have 4*256 MB in link file
        # Use multiple HBMs for history FDMT
        # We can only alloc one single HBM, host can only access one single HBM, but kernel can access multiple HBMs
        # However, we need to make sure that in link file, we assign enough HBM for the FDMT history
        self.fdmt_hist_buf = Buffer((HBM_SIZE), np.int8, device, self.fdmtcu.krnl.group_id(2), self.device_only_buffer_flag).clear() # Grr, group_id puts you in some weird addrss space self.fdmtcu.krnl.group_id(2))
        
        
        # pout of FDMT should be pin of grid reader
        if self.fdmtcu.group_id(1) != self.grid_reader.group_id(0):
            warnings.warn(f'Expected fdmt output to be grid reader input. FDMTout = {self.fdmtcu.group_id(1)} != grid reader { self.grid_reader.group_id(0)}. Perhaps it isnt imporant anymore')

        # Grid reader: pin, ndm, tblk, nchunk, nparallel, axilut, load_luts, streams[4]
        log.info('Allocating mainbuf')

        # Has one DDR in link file, which is 8*1024 MB
        #nt_outbuf = NBLK*self.plan.nt
        #self.mainbuf = Buffer((self.plan.nuvrest, self.plan.ndout, nt_outbuf, self.plan.nuvwide,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()
        mainbuf_shape = (self.plan.nuvrest_max, self.plan.ndout, NBLK, self.plan.nt, self.plan.nuvwide, 2)

        log.info(f'FDMT output buffer size {np.prod(mainbuf_shape)*2/1024/1024/1024} GB')

        num_mainbufs = 8
        sub_mainbuf_shape  = list(mainbuf_shape)
        sub_mainbuf_shape[0] = (self.plan.nuvrest + num_mainbufs - 1) // num_mainbufs
        log.info(f'Mainbuf shape is {mainbuf_shape} breaking into {num_mainbufs} buffers of {sub_mainbuf_shape}')
        # Allocate buffers in sub buffers - this works around an XRT bug that doesn't let you allocate a large buffer
        # every time you allocate a buffer it stacks the address on top of the previous buffer
        self.all_mainbufs = [Buffer(sub_mainbuf_shape, np.int16, device, self.grid_reader.krnl.group_id(0), self.device_only_buffer_flag).clear() for b in range(num_mainbufs)]
            

        log.info('Allocating boxcar_history')    

        npix = self.plan.npix
        # Require 1024 MB, we have 4 HBMs in link file, which gives us 1024 MB
        NBOX = 7 # Xinping only saves 7 boxcars for NBOX = 8. TODO: change to 8
        self.boxcar_history = Buffer((NDM_MAX, NBOX, npix, npix), np.int16, device, self.boxcarcu.group_id(3), self.device_only_buffer_flag).clear() # Grr, gruop_id problem self.boxcarcu.group_id(3))
        log.info(f"Boxcar history {self.boxcar_history.shape} {self.boxcar_history.size} {self.boxcar_history.itemsize}")
        log.info('Allocating candidates')

        # small buffer
        # The buffer size here should match the one declared in C code
        self.candidates = Buffer(NDM_MAX*self.plan.nbox*16, candidate_dtype, device, self.boxcarcu.group_id(5)).clear() # Grrr self.boxcarcu.group_id(3))

        self.starts = None
        self.cal_solution = calibration.CalibrationSolution(self.plan)

        max_fdmt_config_shape = (plan.nuvrest_max, plan.ncin-1, plan.nuvwide, 2)
        self.fdmt_config_buf = Buffer(max_fdmt_config_shape, np.uint16, self.device, self.fdmtcu.krnl.group_id(4)).clear()
        self.grid_luts = [Buffer((GRID_LUT_MAX, ), np.uint16, self.device, g.krnl.group_id(5)).clear() for g in self.grids]
        ddreader_lut_shape = (len(plan.ddreader_lut),)
        self.ddreader_lut = Buffer(ddreader_lut_shape, np.uint32, self.device, self.grid_reader.group_id(5)).clear()
        log.info('CONFIG LUTS: DDREADER shape= %s dtype=%s FDMT: shape=%s dtype=%s GRID: shape=%s dtype=%s',
                 self.ddreader_lut.shape,
                 self.ddreader_lut.dtype,
                 self.fdmt_config_buf.shape,
                 self.fdmt_config_buf.dtype,
                 self.grid_luts[0].shape,
                 self.grid_luts[0].dtype)

        self.subtractor = None
        values = plan.values
        self.flagger = VisFlagger(values.dflag_fradius,
                                  values.dflag_tradius,
                                  values.dflag_cas_threshold,
                                  values.dflag_ics_threshold,
                                  values.dflag_tblk)
        self.update_plan(plan)
        self.image_starts = None
        self.fdmt_starts = None

    def _update_grid_lut(self,plan):
        # If we are on new version of pipeline
        self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, lut = get_grid_lut_from_plan(plan)
        log.info(f'Grid shape {lut.shape} nuv={self.plan.fdmt_plan.nuvtotal}')
        # small buffer
        # For grid with new version pipeline
        # grid LUT shape never changes size, even when nuv changes
        assert lut.shape == self.grid_luts[0].shape, f'Unexpected GRID LUT size. Was: {lut.shape} expected {GRID_LUT_MAX}'
        for l in self.grid_luts:
            l.nparr[:] = lut
            l.copy_to_device()

    def _update_fdmt_lut(self, plan):
        # small buffer
        fdmt_luts = plan.fdmt_plan.fdmt_lut
        assert self.fdmt_config_buf.dtype == fdmt_luts.dtype
        assert self.fdmt_config_buf.shape[1:] == fdmt_luts.shape[1:], f'Unexpected fdmt config buf shape. was {fdmt_luts.shape} expected {self.fdmt_config_buf.shape}'
        assert self.fdmt_config_buf.shape[0] >= fdmt_luts.shape[0], f'Unexpected fdmt config buf shape. was {fdmt_luts.shape[0]} expected {self.fdmt_config_buf.shape[0]}'
        self.fdmt_config_buf.nparr[:] = 0 # reset
        self.fdmt_config_buf.nparr[:fdmt_luts.shape[0],...] = fdmt_luts
        self.fdmt_config_buf.copy_to_device()

    def _update_ddreader_lut(self, plan):
        # DD reader lookup table
        assert plan.ddreader_lut.shape == self.ddreader_lut.shape, f'Unexpected DDREADER lut shape. Was {plan.ddreader_lut.shape} but expected {self.ddreader_lut.shape}'
        self.ddreader_lut.nparr[:] = plan.ddreader_lut
        self.ddreader_lut.copy_to_device()

    def update_plan(self, plan):
        '''
        Copies the lookup tables from the given plan 
        and sets the plan variable.
        Note: you'd better not fiddle with anything like frequencies, or nbl, otherwise we'll have trouble
        '''
        #uv_shape     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide)
        log.info(f"Updating plan to new plan - {plan}")
        self._update_grid_lut(plan)
        self._update_fdmt_lut(plan)
        self._update_ddreader_lut(plan)
        self.fast_baseline2uv = craco.FastBaseline2Uv(plan, conjugate_lower_uvs=True)
        # // reset inbuf to zero so as fast baseline2uv only overwrites where there is data.
        self.inbuf.nparr[:] = 0

        self.plan = plan
        

    @property
    def solarray(self):
        return self.cal_solution.solarray

    @property
    def solarray_avgd(self):
        if self.solarray.ndim == 4:
            return self.solarray.mean(axis=2)
        else:
            return self.solarray
        
    @property
    def num_input_cells(self):
        '''
        Returns the total number of cells added together in the grid
        if we have a calibration array, it's the number of good values in the mask
        Otheerwise, it's the product of nbl, and nf from the plan
        Note: solution array may be dual polarisation, in which case this returns the single polarisation size

        '''
        return self.cal_solution.num_input_cells

    def set_channel_flags(self, chanrange, flagval: bool):
        '''
        sets the channel flags. this is done by updating the calibraiton solution
        '''
        return self.cal_solution.set_channel_flags(chanrange, flagval)
        

    def flag_frequencies_from_file(self, flag_frequency_file:str, flagval:bool):
        '''
        Updates channel flags by loading file and oring in new flags
        '''
        return self.cal_solution.flag_frequencies_from_file(flag_frequency_file, flagval)

    
    def copy_mainbuf(p):
        '''
        Make a copy of the main buffer (hwhich had to be split into pieces due to an XRT limitation)
        Joins it all together and returns a newly allocated buffer
        Takes a while, adn could be huge
        '''
        mainbuf_run = p.all_mainbufs[0]
        main_nuv = mainbuf_run.shape[0]
        mainbuf_shape =list(mainbuf_run.shape[:])
        nbuf = len(p.all_mainbufs)
        mainbuf_shape[0] *= nbuf
        mainbuf = np.zeros(mainbuf_shape, dtype=np.int16)
        for b in range(nbuf):
            start = b*main_nuv
            end = (b+1)*main_nuv
            buf = p.all_mainbufs[b]
            buf.copy_from_device()
            d = buf.nparr
            mainbuf[start:end, ...] = d
            
        return mainbuf

    def run_fdmt(self, tblk):
        '''
        Run the FDMT kernel on the given internal block number and return a KernelStart representing the run
        '''
        assert 0 <= tblk < NBLK
        nuv         = self.plan.fdmt_plan.nuvtotal
        nurest      = nuv//8
        log.info('Running fdmt on tblk=%d nurest=%d', tblk, nurest)
        run = self.fdmtcu(self.inbuf, self.all_mainbufs[0], self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk)
        starts = KernelStarts()
        starts.append(run)
        return starts

    def run_image(self, tblk, values):
        '''
        Run the image pipeline and returna list of waitables
        Clears candidate buffers before executing
        '''
        ndm       = self.plan.nd
        nchunk_time = self.plan.nt//NTIME_PARALLEL
        nuv         = self.plan.fdmt_plan.nuvtotal
        nparallel_uv = nuv//2
        nurest       = nuv//8

        assert nuv % 2 == 0
        assert nuv % 8 == 0
        
        assert nparallel_uv == self.nparallel_uvin, f'The number from pipeline plan should be the same as we calculated based on indexs from pipeline plan. me={nparallel_uv} self={self.nparallel_uvin} nuv={nuv} nurest={nurest}'

        # load lookup tables for ddgrid reader- slows thigns down but do it always for now
        load_luts = 1

        nplane = ndm*nchunk_time
        fft_shift1 = values.fft_shift1 # How much to shift the stage 1 FFT input by
        fft_shift2 = values.fft_shift2 # How much to shift the stage 2 FFT input by
        fft_cfg = (nplane << 16) + (fft_shift2 << 6) + (fft_shift1 << 3)
        fft_cfg2 = make_fft_config(nplane,
                                   fft_shift1,
                                   fft_shift2)
        assert fft_cfg == fft_cfg2, f'Bad fft_cfg calculation. {fft_cfg:x}!={fft_cfg2:x}'

        # scale threshold by target RMS and signal/noise scale
        signal_scale, noise_scale = self.calculate_processing_gain(fft_shift1, fft_shift2)
        threshold = values.threshold
        assert threshold >= 0, f'Invalid threshold:{threshold}'

        img_noiselevel = self.plan.values.target_input_rms*noise_scale
        bc_noiselevel = img_noiselevel / 4 # for some reason BOXCAR values are 4x smaller than image values - check with Xinping
        bc_threshold = threshold*bc_noiselevel
        threshold_boxcarval = np.uint16(bc_threshold)
        self.last_bc_noise_level = bc_noiselevel # save for future reference
        assert threshold_boxcarval > 0, f'Invalid threshold boxcar value {threshold_boxcarval}'

        log.info(f'Configuration just before pipeline running \nndm={ndm} nchunk_time={nchunk_time} tblk={tblk} nuv={nuv} nparallel_uv={nparallel_uv} nurest={nurest} load_luts={load_luts} nplane={nplane} shift1={fft_shift1} shift2={fft_shift2} fft_cfg={fft_cfg:x} threshold={threshold} imgnoise={img_noiselevel} bcnoise={bc_noiselevel} bcthresh={bc_threshold}={threshold_boxcarval}\n')

        assert ndm < 1024,' It hangs for ndm=1024 - not sure why.'

        # need to clear candidates so if there are no candidates before it's run, nothing happens
        self.clear_candidates()
        log.info('Candidates cleared')
        # IT IS VERY IMPORTANT TO START BOXCAR FIRST! IF YOU DON'T THE PIPELINE CAN HANG!
        starts = KernelStarts()
        starts.append(self.boxcarcu(ndm, nchunk_time, threshold_boxcarval, self.boxcar_history, self.boxcar_history, self.candidates))
        
        for cu in self.ffts:
            starts.append(cu(fft_cfg, fft_cfg))
            
        for cu, grid_lut in zip(self.grids, self.grid_luts):
            starts.append(cu(ndm, nchunk_time, self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, grid_lut, load_luts))

        starts.append(self.grid_reader(self.all_mainbufs[0], ndm, tblk, nchunk_time, nurest, self.ddreader_lut, load_luts))

        log.info('%d kernels running', len(starts))
        self.starts = starts
        self.total_retries = 0
        assert starts is not None
        return starts

    def run(self, iblk, values):
        '''
        Runs both FDMT and image pipelines sequentially.
        Not used in proper processing as we want to run then independently on different blocks
        waits for the fdmt to finish but starts and leaves running the image pipeline
        '''

        tblk = iblk % NBLK

        if values.run_fdmt:
            # temporary: finish FDMT before starting image pipeline on same tblk
            #starts.append(self.fdmtcu(self.inbuf, self.mainbuf, self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk))
            # you have to run teh FDMT on a tblk and run the image pipelien on tblk - 1 if you're doing to run them at the same time.
            self.run_fdmt(tblk).wait()

        image_starts = None
        if values.run_image:
            image_starts = self.run_image(tblk, values)
            assert image_starts is not None

        return image_starts


    def wait(self):
        '''
        I'm not sure why I did it this way, rather than waiting on the starts
        Maybe the starts hang for some reason
        '''
        # new XRT doesn't need to poll register
        poll_registers = False
        if poll_registers:
            self.poll_registers()

        if self.starts is not None:
            wait_for_starts(self.starts, self.call_start)
            self.starts = None

    def poll_registers(self):
        '''
        Wait for starts hung with old XRT. We polled registers to work around it.
        WE might not need that anymore
        '''
        log.debug('Sleeping 0.4')
        time.sleep(0.4) # short enough that any reasonable execution will still be running by the time we poll
        for retry in range(1000):
            all_ok = True
            for k in self.all_kernels:
                reg0 = k.krnl.read_register(0x00)
                all_ok &= (reg0 == 0x04)
                log.debug(f'{k.name} reg0={reg0:x} all_ok={all_ok} retry={retry}')

            if all_ok:
                break

            time.sleep(0.01)

        self.total_retries += retry

    def clear_candidates(self):
        '''
        Clear the candidate array
        '''
        # thisis probably a bit extreme, because it sets the whole candidate array to 0, but we
        # We could just set the first entry to zero - I'll work that out later
        self.candidates.clear()
        #assert len(self.get_candidates()) == 0
        

    def get_candidates(self):
        '''
        Returns a numpy array of candidates with candidate_dtype structure type
        Copies all data from the device and returns a view containing only the correct number of valid 
        candidates
        :returns: nparray dtype=candidate_dtype size=number of valid canidates
        '''
        self.candidates.copy_from_device()
        # argmax stops at the first occurence of 'True'
        c = self.candidates.nparr
        if c[-1]['snr'] != 0:
            warnings.warn('Candidate buffer overflowed')
            candout = c
        else:
            ncand = count_candidates(c)
            candout = self.candidates.nparr[:ncand]

        log.info('Got %d candidates. Last candidate is %s', len(candout), c[-1])

        # TODO: think about performance here
        return candout

    def clear_buffers(self, values):
        '''
        Clear main buffer and boxcar and history
        Works whethr we've decieded to map the buffers or not

        For some reason you can't just set the values. But you can run the FDMT 11 times and it will work        '''
        log.info('Clearing mainbuf data NBLK=%s', NBLK)

        if self.alloc_device_only_buffers: # if we have the buffers, we just clear them
            for ibuf, buf in enumerate(self.all_mainbufs):
                buf.clear()
                
            self.inbuf.clear()
            self.fdmt_hist_buf.clear()
            self.boxcar_history.clear()


        else: # If we don't have the bfufers, we have to set the input to 0, and run the pipeline NBLK times
            self.inbuf.clear()
            log.info('Input cleared. Running pipeline')
            for tblk in range(NBLK):
                self.run(tblk, values).wait()

            log.info('Finished clearing pipeline')

    def calibrate_input(self, input_flat_raw):
        '''
        Apply calibration solutions -  Multiply by solution aray
         Need to make a copy, as masked arrays loose the mask if you *= with an unmasked array
        '''
        log.info("Starting calibration")
        if self.solarray is not None:
            # If data is already polsummed, then average the solutions before multiplying
            if self.solarray.ndim == 4 and input_flat_raw.ndim == 3:
                sols = self.solarray.mean(axis=2)
                input_flat = sols*input_flat_raw
            else:
                input_flat = self.solarray*input_flat_raw
        else:
            input_flat = input_flat_raw.copy()

        log.info("Starting normalisation")

        # subtract average over time
        if self.plan.values.subtract >= 0:
            if self.subtractor is None: # TODO: allocate vissubtractor in constructor. It's just tricky to know what the shape will be in advance
                self.subtractor = VisSubtractor(input_flat.shape, self.plan.values.subtract)
                
            input_flat = self.subtractor(input_flat)

        log.info("Starting pol averaging")
        # average polarisations, if necessary
        if input_flat.ndim == 4:
            npol = input_flat.shape[2]
            assert npol == 1 or npol == 2, f'Invalid number of polarisations {npol} {input_flat.shape}'
            if npol == 1:
                input_flat = input_flat[:,:,0,:]
            else:
                input_flat = input_flat.mean(axis=2)

        log.info("Starting rms computation")
        # scale to give target RMS
        targrms = self.plan.values.target_input_rms
        if  targrms > 0:
            # calculate RMS
            real_std = input_flat.real.std()
            imag_std = input_flat.imag.std()
            input_std = np.sqrt(real_std**2+ imag_std**2)/np.sqrt(2) # I think this is right do do quadrature noise over all calibrated data
            # noise over everything
            stdgain = targrms / input_std
            log.info('Input RMS (real/imag) = (%s/%s) quadsum=%s stdgain=%s targetrms=%s', real_std, imag_std, input_std, stdgain, targrms)
            log.info("Applying the rms now")
            input_flat *= stdgain

        return input_flat


    def flag_input(self, input_flat, cas, ics, mask_fil_writer, cas_fil_writer):
        '''
        Update input flagging mask based on running CAS and ICS and IQRM standard deviation
        '''
        return self.flagger(input_flat, cas, ics, mask_fil_writer, cas_fil_writer)

    def calculate_processing_gain(self, fft_shift1, fft_shift2):
        '''
        Calculates the expected signal and noise levels at the images
        returns (signal_gain, noise_gain) as a tuple

        Where signal_gain is the gain applied to a source with a given amplitude per cell/channel in the visibility input
        noise_gain is the RMS in the image to multiply the input noise RMS in the visibilities
        :returns: (signal_gain, noise_gain)
        '''

        nsum = self.num_input_cells
        fft_scale = calc_fft_scale(fft_shift1, fft_shift2)
        signal_gain = nsum*fft_scale
        noise_gain = np.sqrt(nsum)*fft_scale
        return (signal_gain, noise_gain)

    def prepare_inbuf(self, input_flat, values):
        '''
        Converts complex input data in [NBL, NC, *NPOL*, NT] into UV data [NUVWIDE, NCIN, NT, NUVREST]
        Then scales by values.input_scale and NBINARY_POINT_FDMTINPUT 
        Input data can be 3 dimensionsal (assumign NPOL=1) or 4 dimensionsal (assumign NPOL=1 or 2)
        If 4 dimensional, the polarisations are averaged before continuing
        if calibrate is True, calibrates input

        '''
        self.fast_baseline2uv(input_flat, self.inbuf.nparr, values.input_scale)
        return self

    def copy_input(self, input_flat, values):
        '''
        Prepares input buffer then copies to device
        '''
        self.prepare_inbuf(input_flat, values)
        self.inbuf.copy_to_device()

    def copy_and_run_pipeline_parallel(self, iblk, values):
        '''
        Runs everythign in parallel
        Assumes inbuf prepared with prepare_inbuf()
        FDMT runs on iblk in parallel with pipeline on iblk-1
        Returns cand_iblk, candidates - cand_iblk is the block relevatn to the candidates we have.
        Currently cand_iblk = iblk - 2 as thats the depth of the pipeline
        
        :iblk: = input block number. Increments by 1 for every call.
        :returns: cand_iblk, candidates
        '''
        #raise NotImplemented('Theres is probably a bug in this where it outputs empty candidates with S/N=0')

        fdmt_tblk = iblk % NBLK
        img_tblk = (iblk - 1) % NBLK
        cand_iblk = iblk - 2 # candidate block coming from pipeline


        # wait for FDMT
        if self.fdmt_starts is not None:
            self.fdmt_starts.wait()

        # copy input
        self.inbuf.copy_to_device()

        # run fdmt
        self.fdmt_starts = self.run_fdmt(fdmt_tblk)

        # wait for image pipeline
        if self.image_starts is not None:
            self.image_starts.wait()

        # if we've waited successfuly then we can get candidates
        if cand_iblk >= 0:
            candidates = self.get_candidates().copy() # It's about to get cleared - so we should copy it befor eit does
        else:
            candidates = np.zeros(0, dtype=candidate_dtype)

        # only start running once we've run an FDMT
        if iblk >= 1:
            self.image_starts = self.run_image(img_tblk, values)

        log.info('parallel excecution on iblk %d fdmt_tblk=%d img_tblk=%d cand_iblk=%d, ncand=%d', iblk, fdmt_tblk, img_tblk, cand_iblk, len(candidates))

        return cand_iblk, candidates
    
    def copy_and_run_pipeline_serial(self, iblk, values):
        '''
        Runs everything in series
        Assumes inbuf prepared with prepare_inbuf()
        Runs FDMT first, then image pipeline on same buffer.
        Returns cand_iblk, candidates - cand_iblk is the block relevatn to the candidates we have.
        cand_iblk = blk - i.e. no latency.
        
        :iblk: = input block number. Increments by 1 for every call.
        :returns: cand_iblk, candidates
        '''

        # Blocks are sequential
        fdmt_tblk = iblk % NBLK
        img_tblk = iblk % NBLK
        cand_iblk = iblk  # candidate block coming from pipeline


        t = Timer()

        # wait for FDMT
        if self.fdmt_starts is not None:
            self.fdmt_starts.wait()
            t.tick('FDMT init wait')

        # copy input
        self.inbuf.copy_to_device()
        t.tick('Copy to device')

        # run fdmt
        self.fdmt_starts = self.run_fdmt(fdmt_tblk)
        t.tick('FDMT run')
        self.fdmt_starts.wait()
        fdmt_wait_time = t.tick('FDMT wait')
        if fdmt_wait_time.perf < 0.002:
            raise RuntimeError(f'FDMT completed immediately. It should take > 400 ms. There is probably a request stuck in the queue. Maybe reset the card. See CRACO-327. Took: {fdmt_wait_time}')

        # wait for image pipeline
        if self.image_starts is not None:
            self.image_starts.wait()
            t.tick('Image init wait')
        
        self.image_starts = self.run_image(img_tblk, values)
        t.tick('image run')
        self.image_starts.wait()
        t.tick('image wait')

        # if we've waited successfuly then we can get candidates
        if cand_iblk >= 0:
            candidates = self.get_candidates()
            t.tick('Get candidates')
        else:
            candidates = np.zeros(0, dtype=candidate_dtype)
            

        return cand_iblk, candidates
        
def location2pix(location, npix=256):

    npix_half = npix//2
    
    vpix = (location//npix)%npix - npix_half
    if (vpix<0):
        vpix = npix+vpix
        
    upix = location%npix - npix_half
    if (upix<0):
        upix = npix+upix
        
    #location_index = ((npix_half+vpix)%npix)*npix + (npix_half+upix)%npix
    return vpix, upix

location2pix = np.vectorize(location2pix)

def cand2str(candidate, npix, iblk, raw_noise_level):
    location = candidate['loc_2dfft']
    lpix, mpix = location2pix(location, npix)
    rawsn = candidate['snr']
    snr = float(candidate['snr']/raw_noise_level)
    s = f"{snr:.1f}\t{lpix}\t{mpix}\t{candidate['boxc_width']}\t\t{candidate['time']}\t{candidate['dm']}\t{iblk}\t{rawsn}"
    return s

cand_str_header = '# SNR\tlpix\tmpix\tboxc_width\ttime\tdm\tiblk\trawsn\n'
cand_str_wcs_header = cand_str_header[:-1] + "\ttotal_sample\tobstime_sec\tmjd\tdm_pccm3\tra_deg\tdec_deg\n"
    

def print_candidates(candidates, npix, iblk, plan=None):
    print(cand_str_header)
    for candidate in candidates:
        print(cand2str(candidate, npix, iblk))

def cand2str_wcs(c, iblk, plan, first_tstart, raw_noise_level):
    s1 = cand2str(c, plan.npix, iblk, raw_noise_level)
    total_sample = iblk*plan.nt + c['time']
    tsamp_s = plan.tsamp_s
    obstime_sec = total_sample*plan.tsamp_s
    mjd = first_tstart.utc.mjd + obstime_sec.value/3600/24
    dmdelay_ms = c['dm']*tsamp_s.to(units.millisecond)
    dm_pccm3 = dmdelay_ms / DM_CONSTANT / ((plan.fmin/1e9)**-2 - (plan.fmax/1e9)**-2)
    lpix,mpix = location2pix(c['loc_2dfft'], plan.npix)
    coord = plan.wcs.pixel_to_world(lpix, mpix)
    s2 = f'\t{total_sample}\t{obstime_sec.value:0.4f}\t{mjd:0.9f}\t{dm_pccm3.value:0.2f}\t{coord.ra.deg:0.8f}\t{coord.dec.deg:0.6f}'
    return s1+s2

def print_candidates_with_wcs(candidates, iblk, plan, raw_noise_level):
    for c in candidates:
        print(cand2str_wcs(c, iblk, plan, raw_noise_level))

def grid_candidates(cands, field='snr', npix=256):
    g = np.zeros((npix, npix))
    for candidx, cand in enumerate(cands):
        vpix, upix = location2pix(cand['loc_2dfft'], npix)
        if field == 'candidx':
            d = candidx
        elif field == 'count':
            d = 1.0
        else:
            d = cand[field]
        g[vpix, upix] += d

    return g

def waitall(starts):
    for istart, start in enumerate(starts):
        log.info(f'Waiting for istart={istart} start={start}')
        start.wait(0)

def my_mjd(s):
    m = Time(s, scale='tai', format='mjd')
    return m

def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    plan_parser = craft.craco_plan.get_parser()

    parser = ArgumentParser(description='Run search pipeline on a single beam', formatter_class=ArgumentDefaultsHelpFormatter, parents=[plan_parser], conflict_handler='resolve')

    parser.add_argument('-T', '--threshold', action='store', type=float, help='Threshold for pipeline S/N units. Converted to integer when pipeline executed', default=6)
    parser.add_argument('--no-run-fdmt',  action='store_false', dest='run_fdmt', help="Don't FDMT pipeline", default=True)
    parser.add_argument('--no-run-image', action='store_false', dest='run_image', help="Don't Image pipeline", default=True)
    parser.add_argument('--outdir', '-O', help='Directory to write outputs to', default='.')
    parser.add_argument('-w', '--wait',      action='store_true', help='Wait during execution')
    
    parser.add_argument('-b', '--nblocks',   action='store', type=int, help='Number of blocks to process')
    parser.add_argument('-d', '--device',    action='store', type=int, help='Device number')
    parser.add_argument('-x', '--xclbin',    action='store', type=str, help='XCLBIN to load.')
    parser.add_argument('--skip-blocks', type=int, default=0, help='Skip this many bllocks in teh UV file before usign it for UVWs and data')
    parser.add_argument('--start-mjd', type=my_mjd, help='Start MJD (TAI)')
    parser.add_argument('--stop-mjd', type=my_mjd, help='Stop MJD (TAI)')
    parser.add_argument('-s', '--show',      action='store_true',      help='Show plots')
    
    # These three are not used in PipelinePlan ...
    parser.add_argument('-W', '--boxcar_weight', type=str,   help='Boxcar weighting type', choices=('sum','avg','sqrt'), default='sum')
    parser.add_argument('--input-scale', type=float, help='Multiply input by this scale factor before rounding to int16', default=1.0)
    parser.add_argument('-F', '--fft_scale',     type=float, help='Scale FFT output by this amount. If both scales are 1, the output equals the value of frb_amp for crauvfrbsim.py')
    parser.add_argument('--fft-shift1', type=int, help='Shift value for FFT1', default=0)
    parser.add_argument('--fft-shift2', type=int, help='Shift value for FFT2', default=0)
    parser.add_argument('-C','--cand-file', help='Candidate output file txt', default='candidates.txt')
    parser.add_argument('-psf','--save-psf', action='store_true', help='Save psf to disk as fits file every plan_update', default='False')
    parser.add_argument('--dump-mainbufs', type=int, help='Dump main buffer every N blocks', metavar='N')
    parser.add_argument('--dump-fdmt-hist-buf', type=int, help='Dump FDMT history buffer every N blocks', metavar='N')
    parser.add_argument('--dump-boxcar-hist-buf', type=int, help='Dump Boxcar history buffer every N blocks', metavar='N')
    parser.add_argument('--dump-candidates', type=int, help='Dump candidates every N blocks', metavar='N')
    parser.add_argument('--dump-uvdata', type=int, help='Dump input UV data very N blocks', metavar='N')
    parser.add_argument('--dump-input', type=int, help='Dump calibrated baseline data every N blocks', metavar='N')
    parser.add_argument('--show-candidate-grid', choices=('count','candidx','snr','loc_2dfft','boxc_width','time','dm'), help="Show plot of candidates per block")
    parser.add_argument('--injection-file', help='YAML file to use to create injections. If not specified, it will use the data in the FITS file')
    parser.add_argument('--simulate-data', action='store_true', help="Simulate data using the injector instead of reading it from the file. Only to be used in conjunction with the --injection-file option.")
    parser.add_argument('--calibration', help='Calibration .bin file or root of Miriad files to apply calibration')
    parser.add_argument('--target-input-rms', type=float, default=512, help='Target input RMS')
    parser.add_argument('--subtract', type=int, default=256, help='Update subtraction every this number of samples. If <=0 no subtraction will be performed. Must be a multiple of nt or divide evenly into nt')
    parser.add_argument('--flag-ants', type=strrange, help='Ignore these 1-based antenna numbers', default=[])
    parser.add_argument('--flag-chans', help='Flag these channel numbers (strrange)', type=strrange)
    parser.add_argument('--flag-frequency-file', help='Flag channels based on frequency ranges in this file in MHz. One range per line')
    parser.add_argument('--dflag-fradius', help='Dynamic flagging frequency radius. >0 to enable frequency flagging', default=100, type=float)
    parser.add_argument('--dflag-tradius', help='Dynamic flagging time radius. >0 to enable time flagging', default=100, type=float)
    parser.add_argument('--dflag-cas-threshold', help='Dynamic flagging threshold for CAS. >0 to enable CAS flagging', default=5, type=float)
    parser.add_argument('--dflag-ics-threshold', help='Dynamic flagging threshold for ICS. >0 to enable ICS flagging', default=5, type=float)
    parser.add_argument('--dflag-tblk', help='Dynamic flagging block size. Must divide evenly into the block size (256 usually)', default=256, type=int)
    parser.add_argument('--cas-fil', action='store_true', help="Enable saving of the CAS as a filterbank", default=False)
    parser.add_argument('--print-dm0-stats', action='store_true', default=False, help='Print DM0 stats -slows thigns down')
    parser.add_argument('--phase-center-filterbank', default=None, help='Name of filterbank to write phase center data to')
    parser.add_argument('-m','--metadata', help='Path to schedblock metdata .json.gz file')
    parser.add_argument('--update-uv-blocks', help='Update UV coordinates every Nxnt samples (for search_pipeline and MPI pipeline). Set to 0 to disable', type=int, default=6)

    parser.set_defaults(verbose   = False)
    parser.set_defaults(wait      = False)
    parser.set_defaults(show      = False)

    # A lot of these values are fixed at compile time and should be obtained from the firmware
    # - they cant be changed at run time
    
    parser.set_defaults(device    = 0)
    parser.set_defaults(npix      = 256)
    parser.set_defaults(ndm       = 512)
    parser.set_defaults(nt        = 256)
    parser.set_defaults(nbox      = 8)
    parser.set_defaults(nuvwide   = 8)
    parser.set_defaults(nuvmax    = 8192)
    parser.set_defaults(ncin      = 32)
    parser.set_defaults(boxcar_weight = "sum")
    parser.set_defaults(fdmt_scale =1.0)
    parser.set_defaults(fft_scale  =10.0)
    
    parser.set_defaults(os        = "2.1,2.1")
    parser.set_defaults(xclbin    = os.environ.get('XCLBIN'))
    #parser.set_defaults(uv        = "frb_d0_lm0_nt16_nant24.fits")
    #parser.set_defaults(uv        = "frb_d0_t0_a1_sninf_lm00.fits")

    return parser

def do_dump(v, iblk):
    return v is not None and iblk % v == 0


class VisSource:
    def __init__(self, plan, fitsfile, values):
        self.plan = plan
        self.fitsfile = fitsfile
        self.values = values
        if self.values.injection_file is None:
            self.fv = None
            log.info('Reading data from fits %s', self.fitsfile)
        else:
            log.info('Injecting data described by %s', values.injection_file)
            if values.simulate_data:
                vis_source = 'fake'
            else:
                vis_source = None

            self.fv = FakeVisibility(plan, values.injection_file, vis_source = vis_source)

    def __fits_file_iter(self):
        for input_data, uvws in self.fitsfile.fast_time_blocks(self.plan.nt):
            # strip out extra dimensions
            # should now be (nbl, nf, nt)
            input_data = input_data[:,0,0,0,:,0,:]
            
            yield input_data

    def __iter__(self):
        if self.fv is None:
            myiter = self.__fits_file_iter()
        else:
            if self.values.simulate_data:
                myiter = self.fv.gen_fake_blocks()
            else:
                myiter = self.__fits_file_iter()

        return myiter

def open_device(devid, nretry=10,sleep_time=3):
    device = None
    for retry in range(nretry):
        try:
            device = pyxrt.device(devid)
            log.info('Device %s opened successfully on retry %d', devid, retry)
            break
        except:
            log.exception('Could not open device %d. On retry %d. Sleeping for %s seconds', devid, retry, sleep_time)
            if retry == nretry - 1:
                raise
            else:
                time.sleep(sleep_time)

    return device
        
class PipelineWrapper:
    def __init__(self, planinfo, values, devid, startinfo=None, parallel_mode=True, plan=None):
        '''
        Create a pipeilne wrapper
        :planinfo: Adapter containg observation info for the plan. If plan=None a new plan will be created.
        :values: command line arguments
        :device id: pyxrt device ID - I don't recal why this is seaparet
        :startinfo: Info adapter for the beginning of the file - we pull the tstart 
        :parallel_mode: True if you want to run in paralell (default). False for serial.
        from this because the planinfo might have a tstart in the future. If None then we use 
        planinfo
        :plan: Already made plan if we don't have one already
        '''
        if plan is None:
            plan = PipelinePlan(planinfo, values)
            
        self.plan = plan

        # reset dvice first Don't allocate a device becasue I think you get a bus error
        # reset_device(devid) resetting is also problematic. I'm not sure if we want to do that or not. Grrr.
        self.device = open_device(devid)

        self.xbin = pyxrt.xclbin(values.xclbin)
        self.uuid = self.device.load_xclbin(self.xbin)
        self.values = values

        if startinfo is None:
            startinfo = planinfo

        self.parallel_mode = parallel_mode

        self.startinfo = startinfo
        self.first_tstart = startinfo.tstart

        requested_start_mjd = startinfo.tstart if values.start_mjd is None else values.start_mjd
        log.info('Making pipeline wrapper. first_tstart=%s plan_tstart=%s requirested mjd=%s diff=%s', self.first_tstart.tai.mjd,
                 planinfo.tstart.tai.mjd, requested_start_mjd.tai.mjd, (requested_start_mjd - self.first_tstart).to(units.millisecond) )

        # New XRT versions return 'None' for IPs and return kernels instead
        iplist = self.xbin.get_ips()
        if iplist is None:
            iplist = self.xbin.get_kernels()
            
        for iip, ip in enumerate(iplist):
            log.debug('IP %s name=%s', iip, ip.get_name())
                    

        plan = self.plan
        device = self.device
        xbin = self.xbin
        uuid = self.uuid
        beamid = self.plan.beamid

        hdr = {'nbits':32,
               'nchans':plan.nf,
               'nifs':1,
               'src_raj_deg':plan.phase_center.ra.deg,
               'src_dej_deg':plan.phase_center.dec.deg,
               'tstart':self.first_tstart.utc.mjd,
               'tsamp':plan.tsamp_s.value,
               'fch1':plan.fmin/1e6,
               'foff':plan.foff/1e6,
               #'source_name':'UNKNOWN'
        }

        os.makedirs(values.outdir, exist_ok=True)
        self.beamdir = os.path.join(values.outdir, f'beam{beamid:02d}')
        os.makedirs(self.beamdir, exist_ok=True)

        if values.phase_center_filterbank is None:
            self.pc_filterbank = None
        else:
            pcfile = os.path.join(values.outdir, values.phase_center_filterbank.replace('.fil',f'b{beamid:02d}.fil'))
            self.pc_filterbank = sigproc.SigprocFile(pcfile, 'wb', hdr)

        mask_fil_hdr = hdr.copy()
        mask_fil_hdr['nbits'] = 1
        mask_fil_fname = os.path.join(values.outdir, f"RFI_tfmask.b{beamid:02d}.fil")
        self.mask_fil_writer = sigproc.SigprocFile(mask_fil_fname, 'wb', mask_fil_hdr)

        self.cas_fil_writer = None
        if values.cas_fil:
            cas_fil_hdr = hdr.copy()
            cas_fil_hdr['nbits'] = 32
            cas_fil_fname = os.path.join(values.outdir, f"CAS_unnorm.b{beamid:02d}.fil")
            self.cas_fil_writer = sigproc.SigprocFile(cas_fil_fname, 'wb', cas_fil_hdr)
                
        # Create a pipeline
        alloc_device_only = values.dump_mainbufs is not None or \
                            values.dump_fdmt_hist_buf is not None or \
                            values.dump_boxcar_hist_buf is not None or \
                            values.dump_input is not None or \
                            values.print_dm0_stats

        self.doing_dumps = alloc_device_only
    
        p = Pipeline(device, xbin, plan, alloc_device_only)

        if planinfo.freq_config.nmasked_channels > 0:
            log.info('Flagging channels from input: %d', planinfo.freq_config.nmasked_channels)
            p.set_channel_flags(planinfo.freq_config.channel_mask, True)
            
        if values.flag_chans:
            log.info('Flagging %d channels %s from command line', len(values.flag_chans), values.flag_chans)
            p.set_channel_flags(values.flag_chans, True)

        if values.flag_frequency_file:
            log.info('Flagging channels from file %s', values.flag_frequency_file)
            p.flag_frequencies_from_file(values.flag_frequency_file, True)

        self.fixed_freq_weights = ~p.cal_solution.solarray.mask[0, :, 0, 0]
        log.info('Fixed freq weights: %s/%d', self.fixed_freq_weights.sum(), self.fixed_freq_weights.size)

        blk_shape = (plan.nbl, plan.nf, plan.nt)
        self.fast_preprocessor = FastPreprocess(blk_shape, 
                                                p.cal_solution.solarray, 
                                                values, 
                                                self.fixed_freq_weights, 
                                                beamid = beamid,
                                                sky_sub = True, global_norm = True)
        
        tabdir = os.path.join(values.outdir, f'beam{beamid:02d}','tabs')
        target_corods = []
        #self.tab_handler = TAB_handler(target_coords, plan, tabdir)

        self.pipeline = p
        p.clear_buffers(values)

        # make cand file name 'soemthing.b02.txt' and preserve the extension
        cand_file_bits = values.cand_file.split('.')
        cand_file_bits.insert(-1, f'b{beamid:02d}')
        candfile = os.path.join(values.outdir, '.'.join(cand_file_bits))
        candout = CandidateWriter(candfile, plan.freqs, plan.dmax, plan.nbox, self.first_tstart, ibeam=beamid)
        self.total_candidates = 0
        self.candout = candout        
        self.iblk = 0

    def update_plan(self, new_data):
        '''
        Literally make an entire new plan out of the plan info data, and shove int ehlookup tables
        Completley inelegant, but better htan nothing for now
        '''
        log.info('Updating from data')
        old_plan = self.plan
        assert old_plan is not None
        self.plan = PipelinePlan(new_data, self.values, prev_plan=old_plan)
        self.pipeline.update_plan(self.plan)
        return self.plan
    
    def update_plan_from_plan(self, new_plan):
        log.info('Updating plan from plan')
        self.plan = new_plan
        self.pipeline.update_plan(new_plan)
        return self.plan

    def write(self, input_flat, bl_weights=None, input_tf_weights=None, cas=None, ics=None, candout_buffer=None):
        '''
        cas, and ics if specified help with flagging
        '''
        t = Timer()
        self.last_write_timer = t
        p = self.pipeline
        pc_filterbank = self.pc_filterbank
        mask_fil_writer = self.mask_fil_writer
        cas_fil_writer = self.cas_fil_writer
        iblk = self.iblk
        values = self.values
        plan = self.plan

        log.debug("Running block %s input shape=%s dtype=%s", iblk, input_flat.shape, input_flat.dtype)

        if values.simulate_data:
            #Now to make sure that the input data has the desired target input rms I am just going to assume that the simulated data
            #always has an rms of 1 (despite the fact that it is a configurable parameter)

            input_flat_cal = input_flat * values.target_input_rms

        else:
            self.fast_preprocessor(input_flat, bl_weights, input_tf_weights)
            t.tick('preprocessor')
            input_flat_cal = self.fast_preprocessor.output_buf

        if do_dump(values.dump_input, iblk):
            input_flat_cal.dump(os.path.join(self.beamdir, f'input_iblk{iblk}.npy'))# Saves as a pickle load with np.load(allow_pickle=True)
            t.tick('dump input')
        
        if values.injection_file:
            input_flat_cal = self.vis_source.fv.inject_frb_in_data_block(input_flat_cal, iblk, plan)
            t.tick('inject')

        if pc_filterbank is not None:
            d = input_flat_cal.real.mean(axis=0).T.astype(np.float32)
            log.info('Phase center stats %s', printstats(d))
            d.tofile(pc_filterbank.fin)
            pc_filterbank.fin.flush()
            t.tick('PC filterbank')

        p.prepare_inbuf(input_flat_cal, values)
        t.tick('prepare_inbuf')
        
        if self.parallel_mode:
            cand_iblk, candidates = p.copy_and_run_pipeline_parallel(iblk, values)
        else:
            cand_iblk, candidates = p.copy_and_run_pipeline_serial(iblk, values)
        t.tick('run')
        
        log.info('Got %d candidates in block %d cand_iblk=%d', len(candidates), iblk, cand_iblk)
        self.total_candidates += len(candidates)
        out_cands = None

        if self.parallel_mode and self.doing_dumps:
            log.info('Sleeping for 2 seconds as were dumping data but running in parallel mode so we should wait')
            time.sleep(2)

        if len(candidates) > 0:
            out_cands = self.candout.interpret_cands(candidates, cand_iblk, plan, p.last_bc_noise_level, candout_buffer)
            t.tick('Interpret candidates')
            self.candout.write_cands(out_cands)            
            t.tick('Write candidates')

        if values.print_dm0_stats:
            bc = p.boxcar_history.copy_from_device().nparr
            s = 'DM0 image stats' + printstats(bc[0,...]) + f'shape={bc.shape}'
            print(s)
            log.info(s)
            t.tick('Print DM0 stats')

        if len(candidates) > 0 and values.show_candidate_grid is not None:
            img = grid_candidates(candidates, values.show_candidate_grid, npix=256)
            imshow(img, aspect='auto', origin='lower')
            show()
            t.tick('grid candidates')

        # must be after running pipeline otehrwise you copy old data from card into host memory before you dump!
        if do_dump(values.dump_uvdata, iblk): 
            p.inbuf.saveto(os.path.join(self.beamdir, f'uv_data_iblk{iblk}.npy'))
            t.tick('dump uv')

        if do_dump(values.dump_candidates, iblk):
            np.save(f'candidates_iblk{iblk}.npy', candidates) # only save candidates to file - not the whole buffer
            p.candidates.saveto(os.path.join(self.beamdir, f'candidate_buf_iblk{iblk}.npy')) # also save whole buffer because  ... debugging
            t.tick('dump candidates')
        if do_dump(values.dump_mainbufs, iblk):
            for ib, mainbuf in enumerate(p.all_mainbufs):
                mainbuf.saveto(os.path.join(self.beamdir, f'mainbuf_iblk{iblk}_ib{ib}.npy'))

            t.tick('dump mainbuf')

        if do_dump(values.dump_fdmt_hist_buf, iblk):
            p.fdmt_hist_buf.saveto(os.path.join(self.beamdir, f'fdmt_hist_buf_iblk{iblk}.npy'))
            t.tick('dump fdmt hist')

        if do_dump(values.dump_boxcar_hist_buf, iblk):
            p.boxcar_history.saveto(os.path.join(self.beamdir, f'boxcar_hist_iblk{iblk}.npy'))
            t.tick('dump boxcar')

        logging.info('Write for iblk %d timer: %s', iblk, t)

        self.iblk += 1

        return out_cands

    def close(self):
        candout = self.candout
        pc_filterbank = self.pc_filterbank
        mask_fil_writer = self.mask_fil_writer
        cas_fil_writer = self.cas_fil_writer
        values = self.values
        cmdstr =  ' '.join(sys.argv)
        now = datetime.datetime.now()
        logstr = f'# Run {cmdstr} finished on {now}\n'
        candout.write_log(logstr)
        candout.close()
        self.fast_preprocessor.close()
        logging.info('Wrote %s candidates to %s', self.total_candidates, values.cand_file)
        
        if pc_filterbank is not None:
            pc_filterbank.fin.close()

        if mask_fil_writer is not None:
            mask_fil_writer.fin.close()
        
        if cas_fil_writer is not None:
            cas_fil_writer.fin.close()

def _main():
    parser = get_parser()
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log.info(f'Values={values}')

    assert values.max_ndm == NDM_MAX

    # Create a plan
    f = uvfits_meta.open(values.uv, skip_blocks=values.skip_blocks, metadata_file=values.metadata, start_mjd=values.start_mjd, end_mjd=values.stop_mjd)
    f.set_flagants(values.flag_ants)

    update_uv_blocks = values.update_uv_blocks
    nt = values.nt
    if update_uv_blocks == 0:
        isamp_update = 0
    else:
        assert nt % 2 == 0, 'Seems sensible given were about to divide by 2'
        # half way through first block
        isamp_update = update_uv_blocks * nt // 2

    log.info('Creating UVW data for isamp=%d will update every %d samples', isamp_update, update_uv_blocks*nt)
    plan_info = f.vis_metadata(isamp_update)
    start_info = f.vis_metadata(0)
    parallel_mode = True
        
    pipeline_wrapper = PipelineWrapper(plan_info, values, values.device, startinfo=start_info, parallel_mode=parallel_mode)
    plan = pipeline_wrapper.plan
    vis_source = VisSource(plan, f, values)
    pipeline_wrapper.vis_source = vis_source

    if values.wait:
        input('Press any key to continue...')

    t = Timer()
    try:
        for iblk, input_flat in enumerate(vis_source):
            t.tick('read')
            if values.nblocks is not None and iblk >= values.nblocks:
                log.info('Finished due to values.nblocks=%d', values.nblocks)
                break

            if iblk == 0 and values.save_psf:
                psf_name = os.path.join(values.outdir, f"psf.beam{plan.beamid:02g}.iblk{iblk}.fits")
                log.info("Saving the psf to disk with name=%s", psf_name)
                PSF.write_psf(outname=psf_name, plan=plan, iblk=iblk)

            update_now = update_uv_blocks > 0 and iblk % update_uv_blocks == 0 and iblk != 0
            
            if update_now:
                isamp_update += update_uv_blocks*nt
                plan_info = f.vis_metadata(isamp_update)
                t.tick('get_adapter')
                log.info('Updating plan iblk=%d isamp=%d plan_info=%s', iblk, isamp_update, plan_info)
                latest_plan = pipeline_wrapper.update_plan(plan_info)
                if values.save_psf:
                    psf_name = os.path.join(values.outdir, f"psf.beam{latest_plan.beamid:02g}.iblk{iblk}.fits")
                    log.info("Saving the psf to disk with name=%s", psf_name)
                    PSF.write_psf(outname=psf_name, plan=latest_plan, iblk=iblk)
                    plan_fout = os.path.join(values.outdir, f'plan_iblk{iblk}.pkl')
                    latest_plan.save(plan_fout)

            pipeline_wrapper.write(input_flat)
            t.tick('write_data')
            log.info("Read for loop %s", t)
            t = Timer()
    except Exception as E:
        raise E
    finally:
        f.close()
        pipeline_wrapper.close()
                     
if __name__ == '__main__':
    _main()
