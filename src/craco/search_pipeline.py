#!/usr/bin/env python
import numpy as np
from pylab import *
import os
import pyxrt
from .pyxrtutil import *
import time
import pickle
import copy

from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import load_plan

from craft import uvfits
from craft import craco

from Visibility_injector.inject_in_fake_data import FakeVisibility

from collections import OrderedDict

from astropy import units

import logging

DM_CONSTANT = 4.15 # milliseconds and GHz- Shri Kulkarni will kill me

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

NBLK = 11
NCU = 4
NTIME_PARALLEL = (NCU*2)

#NDM_MAX = krnl.MAX_NDM
NDM_MAX = 1024
HBM_SIZE = int(256*1024*1024)
NBINARY_POINT_THRESHOLD = 6
NBINARY_POINT_FDMTIN    = 5


def make_fft_config(nplanes:int,
                    stage1_scale:int = 0,
                    stage2_scale:int = 7,
                    bypass_stage1:bool = False,
                    bypass_transpose:bool = False,
                    bypass_stage2:bool = False) -> int:
    '''
    Generates the FFT config word to execute the FFT.
    The layout of this word I got from unzipping fft2d_v2020.2_rtl_d09062021.xo
    And lookint at TEST_SystyolicFFT2D.vhd and fft2d.v, and the result is
    bit 0 - if 1, bypass stage 1 FFT
    bit 1 - if 1, bypass tranpsose stage
    bit 2 - if 1, bypass stage 2 FFT
    bit 3-5 - shifting to apply at input of the first stage FFT
    bit 6-8 - shifting to apply at input of the 2nd stage FFT
    bits 16-31 - number of planes to process

    @returns FFT config word
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
        super().__init__(device, xbin, 'krnl_ddgrid_reader_4cu:krnl_ddgrid_reader_4cu_1')
  
        
class FfftCu(Kernel):
    def __init__(self, device, xbin, icu):
        super().__init__(device, xbin, f'fft2d:fft2d_{icu+1}')
 
class GridCu(Kernel):
    def __init__(self, device, xbin, icu):
        super().__init__(device, xbin, f'krnl_grid_4cu:krnl_grid_4cu_{icu+1}')

        
class BoxcarCu(Kernel):
    def __init__(self, device, xbin):
        super().__init__(device, xbin, f'krnl_boxc_4cu:krnl_boxc_4cu_1')

class FdmtCu(Kernel):
    def __init__(self, device, xbin):
        super().__init__(device, xbin, 'fdmt_tunable_c32:fdmt_tunable_c32_1')

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
    
    upper_instructions = plan.upper_instructions
    lower_instructions = plan.lower_instructions

    # careful here as craco_plan define nuvmax to be multiple times 8, but we need 2 extra space to pack
    max_nsmp_uv = plan.nuvmax-2
    nuv, nuvout, input_index, output_index, send_marker           = instructions2grid_lut(upper_instructions, max_nsmp_uv)
    h_nuv, h_nuvout, h_input_index, h_output_index, h_send_marker = instructions2grid_lut(lower_instructions, max_nsmp_uv)

    assert nuv == h_nuv # These two should equal

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
        Make the search pipelien bound to the hardware
        
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

        # If we are on new version of pipeline
        self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, lut = get_grid_lut_from_plan(self.plan)
        log.info(f'{self.nparallel_uvin} {self.nparallel_uvout} {self.h_nparallel_uvout}')
        log.info(f'{lut.shape}')
        
        np.savetxt("lut.txt", lut, fmt="%d")
                
        self.grid_reader = DdgridCu(device, xbin)
        self.grids = [GridCu(device, xbin, i) for i in range(4)]
        self.ffts = [FfftCu(device, xbin, i) for i in range(4)]
        self.boxcarcu = BoxcarCu(device, xbin)
        self.fdmtcu = FdmtCu(device, xbin)
        self.all_kernels = [self.grid_reader, self.fdmtcu, self.boxcarcu]
        self.all_kernels.extend(self.grids)
        self.all_kernels.extend(self.ffts)

        log.info(f'lut.shape {lut.shape}')
        log.info(f'nuv {self.plan.fdmt_plan.nuvtotal}')
        
        log.info('Allocating grid LUTs')

        # small buffer
        # For grid with new version pipeline
        self.grid_luts = [Buffer(lut.shape, np.uint16, device, g.krnl.group_id(5)).clear() for g in self.grids]
        
        for l in self.grid_luts:
            l.nparr[:] = lut
            l.copy_to_device()
                
        # FDMT: (pin, pout, histin, histout, pconfig, out_tbkl)
        log.info('Allocating FDMT Input')

        # Used to be like this
        log.info(self.plan.fdmt_plan.nuvtotal)
        log.info(self.plan.nt)
        log.info(self.plan.ncin)
        log.info(self.plan.nuvwide)
        log.info(self.plan.nuvrest)
        log.info(self.plan.ndout)

        #self.plan.fdmt_plan.nuvtotal = 3200
        #self.plan.nuvrest = 400
        
        #log.info(f'{self.plan.fdmt_plan.nuvtotal*self.plan.nt*self.plan.ncin*self.plan.nuvwide*4/1024**2} MB')        
        #log.info(f'{self.plan.nuvrest*self.plan.ndout*NBLK*self.plan.nt*self.plan.nuvwide*4/1024**3} GB')

        #log.info(f'{3200*self.plan.nt*self.plan.ncin*self.plan.nuvwide*4/1024**2} MB')        
        #log.info(f'{400*self.plan.ndout*NBLK*self.plan.nt*self.plan.nuvwide*4/1024**3} GB')

        # Need ??? BM, have 5*256 MB in link file, but it is not device only, can only alloc 256 MB?
        #self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.ncin, self.plan.nt, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()
        #self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.nt, self.plan.ncin, self.plan.nuvwide, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()
        inbuf_shape = (self.plan.nuvrest, self.plan.nt, self.plan.ncin, self.plan.nuvwide, 2)
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
        
        #log.info('Allocating FDMT fdmt_config_buf')
        #self.fdmt_config_buf = Buffer((self.plan.fdmt_plan.nuvtotal*5*self.plan.ncin), np.uint32, device, self.fdmtcu.krnl.group_id(4)).clear()

        # small buffer
        fdmt_luts = self.plan.fdmt_plan.fdmt_lut
        self.fdmt_config_buf = Buffer((fdmt_luts.shape), fdmt_luts.dtype, device, self.fdmtcu.krnl.group_id(4)).clear()
        self.fdmt_config_buf.nparr[:] = fdmt_luts
        self.fdmt_config_buf.copy_to_device()
        
        # pout of FDMT should be pin of grid reader
        assert self.fdmtcu.group_id(1) == self.grid_reader.group_id(0)

        # Grid reader: pin, ndm, tblk, nchunk, nparallel, axilut, load_luts, streams[4]
        log.info('Allocating mainbuf')

        # Has one DDR in link file, which is 8*1024 MB
        #nt_outbuf = NBLK*self.plan.nt
        #self.mainbuf = Buffer((self.plan.nuvrest, self.plan.ndout, nt_outbuf, self.plan.nuvwide,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()
        mainbuf_shape = (self.plan.nuvrest, self.plan.ndout, NBLK, self.plan.nt, self.plan.nuvwide, 2)

        log.info(f'FDMT output buffer size {np.prod(mainbuf_shape)*2/1024/1024/1024} GB')

        num_mainbufs = 8
        sub_mainbuf_shape  = list(mainbuf_shape)
        sub_mainbuf_shape[0] = (self.plan.nuvrest + num_mainbufs - 1) // num_mainbufs
        log.info(f'Mainbuf shape is {mainbuf_shape} breaking into {num_mainbufs} buffers of {sub_mainbuf_shape}')
        # Allocate buffers in sub buffers - this works around an XRT bug that doesn't let you allocate a large buffer
        self.all_mainbufs = [Buffer(sub_mainbuf_shape, np.int16, device, self.grid_reader.krnl.group_id(0), self.device_only_buffer_flag).clear() for b in range(num_mainbufs)]

        # DD reader lookup table
        log.info('Allocating ddreader_lut')
        ddreader_lut_size = (NDM_MAX + self.plan.nuvrest_max) # old version
        ddreader_lut_size = len(plan.ddreader_lut)
        self.ddreader_lut = Buffer(ddreader_lut_size, np.uint32, device, self.grid_reader.group_id(5)).clear()
        self.ddreader_lut.nparr[:] = plan.ddreader_lut
        self.ddreader_lut.copy_to_device()

        log.info('Allocating boxcar_history')    

        npix = self.plan.npix
        # Require 1024 MB, we have 4 HBMs in linke file, which gives us 1024 MB
        NBOX = 7 # Xinping only saves 7 boxcars for NBOX = 8. TODO: change to 8
        self.boxcar_history = Buffer((NDM_MAX, NBOX, npix, npix), np.int16, device, self.boxcarcu.group_id(3), self.device_only_buffer_flag).clear() # Grr, gruop_id problem self.boxcarcu.group_id(3))
        log.info(f"Boxcar history {self.boxcar_history.shape} {self.boxcar_history.size} {self.boxcar_history.itemsize}")
        log.info('Allocating candidates')

        # small buffer
        # The buffer size here should match the one declared in C code
        self.candidates = Buffer(NDM_MAX*self.plan.nbox*16, candidate_dtype, device, self.boxcarcu.group_id(5)).clear() # Grrr self.boxcarcu.group_id(3))

        self.starts = None
        uv_shape     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide)
        self.uv_out  = np.zeros(uv_shape, dtype=np.complex64)
        self.fast_baseline2uv = craco.FastBaseline2Uv(plan, conjugate_lower_uvs=True)


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


    def run(self, blk, values):
        p = self
        if self.starts is not None:
            raise ValueError('ALready started. Call wait()')

        assert self.starts is None
        
        # Should we get the threhsold from values or the plan?
        # Changes dynamically - values
        threshold = values.threshold
        threshold = np.uint16(threshold*(1<<NBINARY_POINT_THRESHOLD))
        ndm       = self.plan.nd

        nchunk_time = self.plan.nt//NTIME_PARALLEL
        tblk = blk % NBLK

        nuv         = self.plan.fdmt_plan.nuvtotal
        nparallel_uv = nuv//2
        nurest       = nuv//8

        assert nuv % 2 == 0
        assert nuv % 8 == 0
        
        assert nparallel_uv == self.nparallel_uvin, 'The number from pipeline plan should be the same as we calculated based on indexs from pipeline plan'

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

        log.info(f'nConfiguration just before pipeline running \nndm={ndm} nchunk_time={nchunk_time} tblk={tblk} nuv={nuv} nparallel_uv={nparallel_uv} nurest={nurest} load_luts={load_luts} nplane={nplane} threshold={threshold} shift1={fft_shift1} shift2={fft_shift2} fft_cfg={fft_cfg:x}\n')

        assert ndm < 1024 # It hangs for ndm=1024 - not sure why.

        starts = []

        self.call_start = time.perf_counter()

        if values.run_fdmt:
            # temporary: finish FDMT before starting image pipeline on same tblk
            #starts.append(self.fdmtcu(self.inbuf, self.mainbuf, self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk))
            # you have to run teh FDMT on a tblk and run the image pipelien on tblk - 1 if you're doing to run them at the same time.
            log.info('Running fdmt')
            self.fdmtcu(self.inbuf, self.all_mainbufs[0], self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk).wait(0)
            log.info('fdmt complete')
        
        if values.run_image:
            # need to clear candidates so if there are no candidates before it's run, nothing happens
            self.clear_candidates()
            log.info('Candidates cleared')
            # IT IS VERY IMPORTANT TO START BOXCAR FIRST! IF YOU DON'T THE PIPELINE CAN HANG!
            starts.append(self.boxcarcu(ndm, nchunk_time, threshold, self.boxcar_history, self.boxcar_history, self.candidates))

            for cu in self.ffts:
                starts.append(cu(fft_cfg, fft_cfg))
            
            for cu, grid_lut in zip(self.grids, self.grid_luts):
                starts.append(cu(ndm, nchunk_time, self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, grid_lut, load_luts))

            starts.append(self.grid_reader(self.all_mainbufs[0], ndm, tblk, nchunk_time, nurest, self.ddreader_lut, load_luts))

        log.info('%d kernels running', len(starts))
        self.starts = starts
        self.total_retries = 0
        return self

    def wait(self):
        for retry in range(10):
            all_ok = True
            log.debug('Sleeping 0.4')
            time.sleep(0.4) # short enough that any reasomable execution will still be running by the time we poll
            for k in self.all_kernels:
                reg0 = k.krnl.read_register(0x00)
                all_ok &= (reg0 == 0x04)
                log.debug(f'{k.name} reg0={reg0:x} all_ok={all_ok} retry={retry}')

            if all_ok:
                break

        self.total_retries += retry

        if self.starts is not None:
            wait_for_starts(self.starts, self.call_start)
            self.starts = None
        

    def clear_candidates(self):
        '''
        Clear the candidate array
        '''
        # thisis probably a bit extreme, because it sets the whole candidate array to 0, but we
        # We could just set the first entry to zero - I'll work that out later
        self.candidates.clear()
        assert len(self.get_candidates()) == 0
        

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
        log.info('Last candidate is %s', c[-1])
        if c[-1]['snr'] != 0:
            warnings.warn('Candidate buffer overflowed')
            candout = c
        else:
            ncand = np.argmax(self.candidates.nparr['snr'] == 0)
            candout = self.candidates.nparr[:ncand]

        # TODO: think about performance here
        return candout.copy()

    def clear_buffers(self, values):
        '''
        Clear main buffer and boxcar and history
        Works wehtehr we've decieded to map the buffers or not

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

    def copy_input(self, input_flat, values):
        '''
        Converts complex input data in [NBL, NC, NT] into UV data [NUVWIDE, NCIN, NT, NUVREST]
        Then scales by values.input_scale and NBINARY_POINT_FDMTINPUT 
        the copies to the device

        '''
        self.fast_baseline2uv(input_flat, self.uv_out)
        self.inbuf.nparr[:,:,:,:,0] = np.round(self.uv_out[:,:,:,:].real*(values.input_scale))
        self.inbuf.nparr[:,:,:,:,1] = np.round(self.uv_out[:,:,:,:].imag*(values.input_scale))
        self.inbuf.copy_to_device()

        return self
        
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

def cand2str(candidate, npix, iblk):
    location = candidate['loc_2dfft']
    lpix, mpix = location2pix(location, npix)
    rawsn = candidate['snr']
    snr = float(candidate['snr'])/float(1<<NBINARY_POINT_THRESHOLD) 
    s = f"{snr:.1f}\t{lpix}\t{mpix}\t{candidate['boxc_width']}\t\t{candidate['time']}\t{candidate['dm']}\t{iblk}\t{rawsn}"
    return s

cand_str_header = '# SNR\tlpix\tmpix\tboxc_width\ttime\tdm\tiblk\trawsn\n'
cand_str_wcs_header = cand_str_header[:-1] + "\ttotal_sample\tobstime_sec\tmjd\tdm_pccm3\tra_deg\tdec_deg\n"
    

def print_candidates(candidates, npix, iblk, plan=None):
    print(cand_str_header)
    for candidate in candidates:
        print(cand2str(candidate, npix, iblk))

def cand2str_wcs(c, iblk, plan):
    s1 = cand2str(c, plan.npix, iblk)
    total_sample = iblk*plan.nt + c['time']
    tsamp_s = plan.tsamp_s
    obstime_sec = total_sample*plan.tsamp_s
    mjd = plan.tstart.mjd + obstime_sec.value/3600/24
    dmdelay_ms = c['dm']*tsamp_s.to(units.millisecond)
    
    dm_pccm3 = dmdelay_ms / DM_CONSTANT / ((plan.fmin/1e9)**-2 - (plan.fmax/1e9)**-2)
    lpix,mpix = location2pix(c['loc_2dfft'], plan.npix)
    coord = plan.wcs.pixel_to_world(lpix, mpix)
    s2 = f'\t{total_sample}\t{obstime_sec.value:0.4f}\t{mjd:0.9f}\t{dm_pccm3.value:0.2f}\t{coord.ra.deg:0.8f}\t{coord.dec.deg:0.6f}'
    return s1+s2

def print_candidates_with_wcs(candidates, iblk, plan):
    for c in candidates:
        print(cand2str_wcs(c, iblk, plan))

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

def wait_for_starts(starts, call_start, timeout=0):
    log.info('Waiting for %d starts', len(starts))
    # I don't know why this helps, but it does, and I don't like it!
    # It was really reliable when it was in there, lets see if its still ok when we remove it.
    #time.sleep(0.1)

    wait_start = time.perf_counter()
    for istart, start in enumerate(starts):
        log.debug(f'Waiting for istart={istart} start={start}')
        start.wait(timeout) # 0 means wait forever
        wait_end = time.perf_counter()
        log.debug(f'Call: {wait_start - call_start} Wait:{wait_end - wait_start}: Total:{wait_end - call_start}')

def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose',   action='store_true', help='Be verbose')
    parser.add_argument('--no-run-fdmt',  action='store_false', dest='run_fdmt', help="Don't FDMT pipeline", default=True)
    parser.add_argument('--no-run-image', action='store_false', dest='run_image', help="Don't Image pipeline", default=True)
    parser.add_argument('-w', '--wait',      action='store_true', help='Wait during execution')
    
    parser.add_argument('-b', '--nblocks',   action='store', type=int, help='Number of blocks to process')
    parser.add_argument('-d', '--device',    action='store', type=int, help='Device number')
    #parser.add_argument('-n', '--npix',      action='store', type=int, help='Number of pixels in image')
    parser.add_argument('-c', '--cell',      action='store', type=int, help='Image cell size (arcsec). Overrides --os')
    parser.add_argument('-m', '--ndm',       action='store', type=int, help='Number of DM trials')
    #parser.add_argument('--max-ndm', help='Maximum number of DM trials. MUST AGREE WITH FIRMWARE - DO NOT CHANGE UNLESS YOU KNW WHAT YOUR DOING', type=int, default=1024)
    #parser.add_argument('-t', '--nt',        action='store', type=int, help='Number of times per block')
    #parser.add_argument('-B', '--nbox',      action='store', type=int, help='Number of boxcar trials')
    #xparser.add_argument('-U', '--nuvwide',   action='store', type=int, help='Number of UV processed in parallel')
    #parser.add_argument('-N', '--nuvmax',    action='store', type=int, help='Maximum number of UV allowed.')
    #parser.add_argument('-C', '--ncin',      action='store', type=int, help='Numer of channels for sub fdmt')
    #parser.add_argument('-D', '--ndout',     action='store', type=int, help='Number of DM for sub fdmt')
    
    parser.add_argument('-T', '--threshold', action='store', type=float, help='Threshold for pipeline S/N units. Converted to integer when pipeline executed')
    parser.add_argument('-o', '--os',        action='store', type=str, help='Number of pixels per beam')
    
    parser.add_argument('-x', '--xclbin',    action='store', type=str, help='XCLBIN to load.')
    parser.add_argument('-u', '--uv',        action='store', type=str, help='Load antenna UVW coordinates from this UV file')
    parser.add_argument('-s', '--show',      action='store_true',      help='Show plots')
    
    # These three are not used in PipelinePlan ...
    parser.add_argument('-W', '--boxcar_weight', type=str,   help='Boxcar weighting type', choices=('sum','avg','sqrt'), default='sum')
    parser.add_argument('--input-scale', type=float, help='Multiply input by this scale factor before rounding to int16', default=1.0)
    parser.add_argument('-F', '--fft_scale',     type=float, help='Scale FFT output by this amount. If both scales are 1, the output equals the value of frb_amp for crauvfrbsim.py')
    parser.add_argument('--fft-shift1', type=int, help='Shift value for FFT1', default=0)
    parser.add_argument('--fft-shift2', type=int, help='Shift value for FFT2', default=7)
    parser.add_argument('-C','--cand-file', help='Candidate output file txt', default='candidates.txt')
    parser.add_argument('--dump-mainbufs', type=int, help='Dump main buffer every N blocks', metavar='N')
    parser.add_argument('--dump-fdmt-hist-buf', type=int, help='Dump FDMT history buffer every N blocks', metavar='N')
    parser.add_argument('--dump-boxcar-hist-buf', type=int, help='Dump Boxcar history buffer every N blocks', metavar='N')
    parser.add_argument('--dump-candidates', type=int, help='Dump candidates every N blocks', metavar='N')
    parser.add_argument('--dump-uvdata', type=int, help='Dump input UV data every N blocks', metavar='N')
    parser.add_argument('--show-candidate-grid', choices=('count','candidx','snr','loc_2dfft','boxc_width','time','dm'), help="Show plot of candidates per block")
    parser.add_argument('--injection-file', help='YAML file to use to create injections. If not specified, it will use the data in the FITS file')
    
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
    parser.set_defaults(ndout     = 186) # used to be 32
    parser.set_defaults(threshold = 10.0)
    parser.set_defaults(boxcar_weight = "sum")
    parser.set_defaults(fdmt_scale =1.0)
    parser.set_defaults(fft_scale  =10.0)
    
    parser.set_defaults(os        = "2.1,2.1")
    parser.set_defaults(xclbin    = "binary_container_1.xclbin.golden")
    parser.set_defaults(uv        = "frb_d0_lm0_nt16_nant24.fits")
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
            log.info('Injectiing data from fits %s', self.fitsfile)
        else:
            log.info('Injecting data described by %s', values.injection_file)
            self.fv = FakeVisibility(plan, values.injection_file, int(1e6))

    def __fits_file_iter(self):
        for input_data in self.fitsfile.time_blocks(self.plan.nt):
            input_flat = craco.bl2array(input_data) # convert to dictionary - ordered by baseline
            yield input_flat

    def __iter__(self):
        if self.fv is None:
            myiter = self.__fits_file_iter()
        else:
            myiter = self.fv.get_fake_data_block()

        return myiter

def _main():
    parser = get_parser()
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log.info(f'Values={values}')

    mode   = get_mode()
    device = pyxrt.device(values.device)
    xbin = pyxrt.xclbin(values.xclbin)
    uuid = device.load_xclbin(xbin)
    iplist = xbin.get_ips()
    for ip in iplist:
        log.info(ip.get_name())

    # Create a plan
    f = uvfits.open(values.uv)
    plan = PipelinePlan(f, values)

    # Create a pipeline
    alloc_device_only = values.dump_mainbufs is not None or \
                        values.dump_fdmt_hist_buf is not None or \
                        values.dump_boxcar_hist_buf is not None
    
    p = Pipeline(device, xbin, plan, alloc_device_only)

    uv_shape     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide)
    uv_shape2     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide, 2)
    uv_out  = np.zeros(uv_shape, dtype=np.complex64)
    uv_out_fixed = np.zeros(uv_shape2, dtype=np.int16)

    vis_source = VisSource(plan, f, values)

    if values.wait:
        input('Press any key to continue...')

    p.clear_buffers(values)
    
    candout = open(values.cand_file, 'w')
    candout.write(cand_str_wcs_header)
    total_candidates = 0
    bestcand = None

    for iblk, input_flat in enumerate(vis_source):
        if values.nblocks is not None and iblk >= values.nblocks:
            log.info('Finished due to values.nblocks=%d', values.nblocks)
            break

        log.debug("Running block %s", iblk)
        p.copy_input(input_flat, values) # take the input into the device
        
        if do_dump(values.dump_uvdata, iblk):
            p.inbuf.saveto(f'uv_data_iblk{iblk}.npy')

        p.run(iblk, values).wait() # Run pipeline
        
        candidates = p.get_candidates().copy()

        log.info('Got %d candidates in block %d', len(candidates), iblk)
        total_candidates += len(candidates)
        for c in candidates:
            candout.write(cand2str_wcs(c, iblk, plan)+'\n')

        if len(candidates) > 0 and values.show_candidate_grid is not None:
            img = grid_candidates(candidates, values.show_candidate_grid, npix=256)
            imshow(img, aspect='auto', origin='lower')
            show()

        if do_dump(values.dump_candidates, iblk):
            np.save(f'candidates_iblk{iblk}.npy', candidates) # only save candidates to file - not the whole buffer
        if do_dump(values.dump_mainbufs, iblk):
            for ib, mainbuf in enumerate(p.all_mainbufs):
                mainbuf.saveto(f'mainbuf_iblk{iblk}_ib{ib}.npy')

        if do_dump(values.dump_fdmt_hist_buf, iblk):
            p.fdmt_hist_buf.saveto(f'fdmt_hist_buf_iblk{iblk}.npy')

        if do_dump(values.dump_boxcar_hist_buf, iblk):
            p.boxcar_history.saveto(f'boxcar_hist_iblk{iblk}.npy')
                               

    f.close()

    cmdstr =  ' '.join(sys.argv)
    now = datetime.datetime.now()
    logstr = f'# Run {cmdstr} finished on {now}\n'
    candout.write(logstr)
    candout.flush()    
    candout.close()
    logging.info('Wrote %s candidates to %s', total_candidates, values.cand_file)

                     
if __name__ == '__main__':
    _main()
