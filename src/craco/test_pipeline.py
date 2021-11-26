#!/usr/bin/env python
import numpy as np
from pylab import *
import os
import pyxrt
from craco_testing.pyxrtutil import *
import time
import pickle
from craft.craco_plan import PipelinePlan
from craft.craco_plan import FdmtPlan
from craft.craco_plan import FdmtRun
from craft.craco_plan import load_plan

'''
we have a lot of hard-code number here, which is very dangerous
'''

def get_mode():
    mode = os.environ.get('XCL_EMULATION_MODE', 'hw')
    return mode

class AddInstruction(object):
    def __init__(self, plan, target_slot, cell_coords, uvpix):
        self.plan = plan
        self.target_slot = target_slot
        self.cell_coords = cell_coords
        self.uvpix = uvpix
        self.shift = False

    @property
    def shift_flag(self):
        return 1 if self.shift else 0

    @property
    def uvidx(self):
        irun, icell = self.cell_coords
        c = icell + self.plan.nuvwide*irun
        return c

    def __str__(self):
        irun, icell = self.cell_coords
        cell = self.plan.fdmt_plan.get_cell(self.cell_coords)
        return 'add {self.cell_coords} which is {cell} to slot {self.target_slot} and shift={self.shift}'.format(self=self, cell=cell)

    __repr__ = __str__
    
#NDOUT = 186 # self.plan.ndout
#NT    = 256 # self.plan.nt
#NCIN  = 32  # self.plan.ncin
#NUVWIDE = 8  # self.plan.nuvwide
#NT_OUTBUF = NBLK*NT
#NUV     = 4800 # ???
#NUVWIDE = 8
#NUREST  = NUV // NUVWIDE

NBLK  = 3
NCU = 4
NTIME_PARALLEL = (NCU*2)

NDM_MAX = 1024
NPIX = 256
NSMP_2DFFT  = (NPIX*NPIX)
MAX_NSMP_UV = 8190 # This should match the number in pipeline krnl.hpp file
#MAX_NSMP_UV = 4800 # This should match the number in pipeline krnl.hpp file
MAX_NPARALLEL_UV = (MAX_NSMP_UV//2)

NEW_GRID = True
#NEW_GRID = False

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

def instructions2grid_lut(instructions):
    data = np.array([[i.target_slot, i.uvidx, i.shift_flag, i.uvpix[0], i.uvpix[1]] for i in instructions], dtype=np.int32)
    
    nuvout = len(data[:,0]) # This is the number of output UV, there are zeros in between

    output_index = data[:,0]
    input_index  = data[:,1]
    send_marker  = data[:,2][1::2]

    nuvin = np.sort(input_index)[-1]   # This is the real number of input UV
    
    output_index_hw = np.pad(output_index, (0, MAX_NSMP_UV-nuvout), 'constant')
    input_index_hw  = np.pad(input_index,  (0, MAX_NSMP_UV-nuvout), 'constant')
    send_marker_hw  = np.pad(send_marker,  (0, MAX_NPARALLEL_UV-int(nuvout//2)), 'constant')
    
    return nuvin, nuvout, input_index_hw, output_index_hw, send_marker_hw

def instructions2pad_lut(instructions):
    location = np.zeros(NSMP_2DFFT, dtype=int)

    data = np.array(instructions, dtype=np.int32)
    upix = data[:,0]
    vpix = data[:,1]

    location_index = vpix*NPIX+upix    
    location_value = data[:,2]+1

    location[location_index] = location_value

    return location
    
def get_grid_lut_from_plan(plan):
    
    upper_instructions = plan.upper_instructions
    lower_instructions = plan.lower_instructions
    
    nuv, nuvout, input_index, output_index, send_marker           = instructions2grid_lut(upper_instructions)
    h_nuv, h_nuvout, h_input_index, h_output_index, h_send_marker = instructions2grid_lut(lower_instructions)

    assert nuv == h_nuv # These two should equal

    nuv_round = nuv+(8-nuv%8)       # Round to 8
    assert nuv_round <= MAX_NSMP_UV # WE can not go above MAX_NSMP_UV

    location   = instructions2pad_lut(plan.upper_idxs)
    h_location = instructions2pad_lut(plan.lower_idxs)
    
    shift_marker   = np.array(plan.upper_shifts, dtype=np.uint16)
    h_shift_marker = np.array(plan.lower_shifts, dtype=np.uint16)
    
    lut = np.concatenate((output_index, input_index, send_marker, location, shift_marker, h_output_index, h_input_index, h_send_marker, h_location, h_shift_marker)).astype(np.uint16)
    
    return nuv_round//2, nuvout//2, h_nuvout//2, lut
    
class Pipeline:
    def __init__(self, device, xbin, plan_fname):
        self.plan = load_plan(plan_fname)

        if NEW_GRID:
            # If we are on new version of pipeline
            self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, lut = get_grid_lut_from_plan(self.plan)
            print(f'{self.nparallel_uvin} {self.nparallel_uvout} {self.h_nparallel_uvout}')
            print(f'{lut.shape}')
            
            np.savetxt("lut.txt", lut, fmt="%d")
            #np.savetxt("lut.txt", lut)
            #exit()
        else:
            # if we are on old version of pipeline, which grid does not have accumulation function
            lutbin = 'none_duplicate_long.uvgrid.txt.bin'
            print(f'Using lut binary file {lutbin}')
            lut = np.fromfile(lutbin, dtype=np.uint32)
            print(f'LUT size is {len(lut)}')
        
        self.grid_reader = DdgridCu(device, xbin)
        self.grids = [GridCu(device, xbin, i) for i in range(4)]
        self.ffts = [FfftCu(device, xbin, i) for i in range(4)]
        self.boxcarcu = BoxcarCu(device, xbin)
        self.fdmtcu = FdmtCu(device, xbin)

        print(f'lut.shape {lut.shape}')
        print(f'nuv {self.plan.fdmt_plan.nuvtotal}')
        
        print('Allocating grid LUTs')
        if NEW_GRID:
            # For grid with new version pipeline
            self.grid_luts = [Buffer(lut.shape, np.uint16, device, g.krnl.group_id(5)).clear() for g in self.grids]
        else:
            # For grid with old version pipeline
            self.grid_luts = [Buffer(lut.shape, np.uint32, device, g.krnl.group_id(3)).clear() for g in self.grids]

        for l in self.grid_luts:
            l.nparr[:] = lut
            l.copy_to_device()
                
        # FDMT: (pin, pout, histin, histout, pconfig, out_tbkl)
        print('Allocating FDMT Input')

        self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.ncin, self.plan.nt, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()        
                
        # FDMT histin, histhout should be same buffer
        assert self.fdmtcu.group_id(2) == self.fdmtcu.group_id(3), 'FDMT histin and histout should be the same'
        
        print('Allocating FDMT history')
        self.fdmt_hist_buf = Buffer((256*1024*1024), np.int8, device, self.fdmtcu.krnl.group_id(2), 'device_only').clear() # Grr, group_id puts you in some weird addrss space self.fdmtcu.krnl.group_id(2))
        
        print('Allocating FDMT fdmt_config_buf')
        #self.fdmt_config_buf = Buffer((self.plan.fdmt_plan.nuvtotal*5*self.plan.ncin), np.uint32, device, self.fdmtcu.krnl.group_id(4)).clear()
        
        fdmt_luts = self.plan.fdmt_plan.fdmt_lut
        self.fdmt_config_buf = Buffer((fdmt_luts.shape), fdmt_luts.dtype, device, self.fdmtcu.krnl.group_id(4)).clear()
        self.fdmt_config_buf.nparr[:] = fdmt_luts
        self.fdmt_config_buf.copy_to_device()
        
        # pout of FDMT should be pin of grid reader
        assert self.fdmtcu.group_id(1) == self.grid_reader.group_id(0)

        # Grid reader: pin, ndm, tblk, nchunk, nparallel, axilut, load_luts, streams[4]
        print('Allocating mainbuf')
        nt_outbuf = NBLK*self.plan.nt
        # WE should NOT use self.plan.nuvrest here, we need to calculate nuvrest as follow
        nuvrest = self.plan.fdmt_plan.nuvtotal//self.plan.nuvwide
        self.mainbuf = Buffer((nuvrest, self.plan.ndout, nt_outbuf, self.plan.nuvwide,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()

        print('Allocating ddreader_lut')
        self.ddreader_lut = Buffer((NDM_MAX + nuvrest), np.uint32, device, self.grid_reader.group_id(5)).clear()
        print('Allocating boxcar_history')    
        self.boxcar_history = Buffer((NDM_MAX, NPIX, NPIX, 2), np.int16, device, self.boxcarcu.group_id(3), 'device_only').clear() # Grr, gruop_id problem self.boxcarcu.group_id(3))
        print('Allocating candidates')    
        self.candidates = Buffer(256*1024*1024, np.int8, device, self.boxcarcu.group_id(5)).clear() # Grrr self.boxcarcu.group_id(3))


def run(p, blk, values):
    self = p
    threshold = values.threshold
    #ndm = values.ndm
    ndm = self.plan.nd
    nchunk_time = self.plan.nt//NTIME_PARALLEL

    #nchunk_time = values.nchunk_time
    tblk = (values.tblk + blk ) % NBLK
    #nuv = values.nuv
    nuv = self.plan.fdmt_plan.nuvtotal
    
    nparallel_uv = nuv//2
    nurest = nuv//8
    load_luts = 1

    nplane = ndm*nchunk_time
    shift1 = 0 # FFT CONFIG register - not sure what this means
    shift2 = 7 # FFT CONFIG Register - not sure what this means
    fft_cfg = (nplane << 16) + (shift2 << 6) + (shift1 << 3)

    print(f'\nConfiguration just before pipeline running \nndm={ndm} nchunk_time={nchunk_time} tblk={tblk} nuv={nuv} nparallel_uv={nparallel_uv} nurest={nurest} load_luts={load_luts} nplane={nplane} shift1={shift1} shift2={shift2} fft_cfg={fft_cfg}\n')

    #values.run_pipeline = False #True
    values.run_pipeline = True
    values.run_fdmt     = False

    assert ndm < 1024 # It hangs for 1024 - not sure why.

    starts = []
    
    if NEW_GRID:
        assert nparallel_uv == self.nparallel_uvin # the number from pipeline plan should be the same as we calculated based on indexs from pipeline plan
        
    if values.run_pipeline:
        #assert nuv == 3440 # NUV and the LUT need to agree - if not you get in trouble
        for cu in self.ffts:
            starts.append(cu(fft_cfg, fft_cfg))
            
        starts.append(self.boxcarcu(ndm, nchunk_time, threshold, self.boxcar_history, self.boxcar_history, self.candidates))
        starts.append(self.grid_reader(self.mainbuf, ndm, tblk, nchunk_time, nurest, self.ddreader_lut, load_luts))

        for cu, grid_lut in zip(self.grids, self.grid_luts):
            if NEW_GRID:
                # For grid with new pipeline
                starts.append(cu(ndm, nchunk_time, self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, grid_lut, load_luts))
            else:
                # For grid with old pipeline
                starts.append(cu(ndm, nchunk_time, nparallel_uv, grid_lut, load_luts))

    if values.run_fdmt:
        starts.append(self.fdmtcu(self.inbuf, self.mainbuf, self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk))

    return starts


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-b','--nblocks', default=1, type=int, help='Number of blocks')
    parser.add_argument('-u','--nuv', type=int, help='Number of NUV must match LUTS otherwise lockup', default=3440)
    parser.add_argument('-m','--ndm', default=1, type=int, help='Number of DMs')
    parser.add_argument('-t','--threshold', default=1, type=int, help='Threshold for boxcar')
    parser.add_argument('-c','--nchunk-time', default=32, type=int, help='Nchunks of time to do')
    parser.add_argument('-k','--tblk', default=0, type=int, help='Block number to execute')
    parser.add_argument('--no-fdmt', default=True, action='store_false', help='Dont run FDMT pipeline', dest='run_fdmt')
    parser.add_argument('--no-image', default=True, action='store_false', help='Dont run Image pipeline', dest='run_pipeline')
    parser.add_argument('-x', '--xclbin', default=None, help='XCLBIN to load.', required=False)
    parser.add_argument('-d','--device', default=0, type=int,help='Device number')
    parser.add_argument('--wait', default=False, action='store_true', help='Wait during execution')
    parser.add_argument('-p', '--plan', type=str, action='store', help='plan file which has pipeline configurations')

    parser.set_defaults(verbose=False)
    if NEW_GRID:
        parser.set_defaults(xclbin="binary_container_1.xclbin.CRACO-46")
        parser.set_defaults(plan="pipeline.pickle")
    else:
        parser.set_defaults(xclbin="binary_container_1.xclbin.CRACO-42")
        parser.set_defaults(plan="pipeline_short.pickle")

    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print(f'Values={values}')


    mode   = get_mode()
    device = pyxrt.device(0)
    xbin = pyxrt.xclbin(values.xclbin)
    uuid = device.load_xclbin(xbin)
    iplist = xbin.get_ips()
    for ip in iplist:
        print(ip.get_name())

    p = Pipeline(device, xbin, values.plan)
    
    p.inbuf.nparr[:] = 1
    p.inbuf.copy_to_device()

    if values.wait:
        input('Press any key to continue...')
        
    for blk in range(values.nblocks):
        call_start = time.perf_counter()
        starts = run(p, blk, values)
        wait_start = time.perf_counter()
    
        for istart, start in enumerate(starts):
            print(f'Waiting for istart={istart} start={start}')
            start.wait(0)

            wait_end = time.perf_counter()
            print(f'Call: {wait_start - call_start} Wait:{wait_end - wait_start}: Total:{wait_end - call_start}')
            
    #print(values)

    p.mainbuf.copy_from_device()
    print(p.mainbuf.nparr.shape)

    p.candidates.copy_from_device()
    print(np.all(p.candidates.nparr == 0))
    p.boxcar_history.copy_from_device()
    print(np.all(p.boxcar_history.nparr == 0))

    p.fdmt_hist_buf.copy_to_device()
    print('inbuf', hex(p.inbuf.buf.address()))
    print('mainbuf', hex(p.mainbuf.buf.address()))
    print('histbuf', hex(p.fdmt_hist_buf.buf.address()))
    print('fdmt_config_buf', hex(p.fdmt_config_buf.buf.address()))

    print(f'{p.plan.fdmt_plan.nuvtotal}')
    

if __name__ == '__main__':
    _main()
