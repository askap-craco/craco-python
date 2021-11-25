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

NDOUT = 186
NT = 256
NBLK = 3
NT_OUTBUF = NBLK*NT
NCIN = 32
NUV = 4800
NUVWIDE = 8
NUREST = NUV // NUVWIDE
NDM_MAX = 1024
NPIX = 256
NSMP_2DFFT = (NPIX*NPIX)

MAX_NSMP_UV = 8190 # This should match the number in pipeline krnl.hpp file
MAX_NPARALLEL_UV = (MAX_NSMP_UV//2)

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
    
    nuv = len(data[:,0])

    output_index = data[:,0]
    input_index  = data[:,1]
    send_marker  = data[:,2][1::2]

    output_index_hw = np.pad(output_index, (0, MAX_NSMP_UV-nuv), 'constant')
    input_index_hw  = np.pad(input_index,  (0, MAX_NSMP_UV-nuv), 'constant')
    send_marker_hw  = np.pad(send_marker,  (0, MAX_NPARALLEL_UV-int(nuv//2)), 'constant')
    
    return input_index_hw, output_index_hw, send_marker_hw

def instructions2pad_lut(instructions):
    location = np.zeros(NSMP_2DFFT, dtype=int)

    data = np.array(instructions, dtype=np.int32)
    upix = data[:,0]
    vpix = data[:,1]

    location_index = vpix*NPIX+upix    
    location_value = data[:,2]+1

    location[location_index] = location_value

    return location
    

class Pipeline:
    def __init__(self, device, xbin, plan_fname):
        self.plan = load_plan(plan_fname)
        self.upper_instructions = self.plan.upper_instructions
        self.lower_instructions = self.plan.lower_instructions

        self.input_index, self.output_index, self.send_marker       = instructions2grid_lut(self.upper_instructions)
        self.h_input_index, self.h_output_index, self.h_send_marker = instructions2grid_lut(self.lower_instructions)

        self.location   = instructions2pad_lut(self.plan.upper_idxs)
        self.h_location = instructions2pad_lut(self.plan.lower_idxs)
        
        self.shift_marker   = np.array(self.plan.upper_shifts, dtype=np.int32)
        self.h_shift_marker = np.array(self.plan.lower_shifts, dtype=np.int32)
        exit()
        
        self.grid_reader = DdgridCu(device, xbin)
        self.grids = [GridCu(device, xbin, i) for i in range(4)]
        self.ffts = [FfftCu(device, xbin, i) for i in range(4)]
        self.boxcarcu = BoxcarCu(device, xbin)
        self.fdmtcu = FdmtCu(device, xbin)
        
        print('Allocating grid LUTs')
        self.grid_luts = [Buffer(lut.shape, np.uint32, device, g.group_id(3)).clear() for g in self.grids]
        for l in self.grid_luts:
            l.nparr[:] = lut
            l.copy_to_device()
        
        # FDMT: (pin, pout, histin, histout, pconfig, out_tbkl)
        print('Allocating FDMT Input')

        self.inbuf = Buffer((NUV, NCIN, NT, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()        
                
        # FDMT histin, histhout should be same buffer
        assert self.fdmtcu.group_id(2) == self.fdmtcu.group_id(3), 'FDMT histin and histout should be the same'
        
        print('Allocating FDMT history')
        self.fdmt_hist_buf = Buffer((256*1024*1024), np.int8, device, self.fdmtcu.krnl.group_id(2), 'device_only').clear() # Grr, group_id puts you in some weird addrss space self.fdmtcu.krnl.group_id(2))
        
        print('Allocating FDMT fdmt_config_buf')
        self.fdmt_config_buf = Buffer((NUV*5*NCIN), np.uint32, device, self.fdmtcu.krnl.group_id(4)).clear()

        # pout of FDMT should be pin of grid reader
        assert self.fdmtcu.group_id(1) == self.grid_reader.group_id(0)

        # Grid reader: pin, ndm, tblk, nchunk, nparallel, axilut, load_luts, streams[4]
        print('Allocating mainbuf')
        self.mainbuf = Buffer((NUREST, NDOUT, NT_OUTBUF, NUVWIDE,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()

        print('Allocating ddreader_lut')
        self.ddreader_lut = Buffer((NDM_MAX + NUREST), np.uint32, device, self.grid_reader.group_id(5)).clear()
        print('Allocating boxcar_history')    
        self.boxcar_history = Buffer((NDM_MAX, NPIX, NPIX, 2), np.int16, device, self.boxcarcu.group_id(3), 'device_only').clear() # Grr, gruop_id problem self.boxcarcu.group_id(3))
        print('Allocating candidates')    
        self.candidates = Buffer(256*1024*1024, np.int8, device, self.boxcarcu.group_id(5)).clear() # Grrr self.boxcarcu.group_id(3))


def run(p, blk, values):
    self = p
    threshold = values.threshold
    ndm = values.ndm
    nchunk_time = values.nchunk_time
    tblk = (values.tblk + blk ) % NBLK
    nuv = values.nuv
    nparallel_uv = nuv//2
    nurest = nuv//8
    load_luts = 1

    nplane = ndm*nchunk_time
    shift1 = 0 # FFT CONFIG register - not sure what this means
    shift2 = 7 # FFT CONFIG Register - not sure what this means
    fft_cfg = (nplane << 16) + (shift2 << 6) + (shift1 << 3)

    print(f'ndm={ndm} nchunk_time={nchunk_time} tblk={tblk} nuv={nuv} nparallel_uv={nparallel_uv} nurest={nurest} load_luts={load_luts} nplane={nplane} shift1={shift1} shift2={shift2} fft_cfg={fft_cfg}')
    run_pipeline = True
    run_fdmt = True

    assert ndm < 1024 # It hangs for 1024 - not sure why.

    starts = []

    if values.run_pipeline:
        assert nuv == 3440 # NUV and the LUT need to agree - if not you get in trouble
        for cu in self.ffts:
            starts.append(cu(fft_cfg, fft_cfg))
            
        starts.append(self.boxcarcu(ndm, nchunk_time, threshold, self.boxcar_history, self.boxcar_history, self.candidates))
        starts.append(self.grid_reader(self.mainbuf, ndm, tblk, nchunk_time, nurest, self.ddreader_lut, load_luts))

        for cu, grid_lut in zip(self.grids, self.grid_luts):
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
    parser.add_argument('-e', '--version', default='', help='Version of fw to load. e.g. ".v14"')
    parser.add_argument('-x', '--xclbin', default=None, help='XCLBIN to load. Overrides version', required=False)
    parser.add_argument('-d','--device', default=0, type=int,help='Device number')
    parser.add_argument('--wait', default=False, action='store_true', help='Wait during execution')
    parser.add_argument('-p', '--plan', default='pipeline.pickle', type=str, action='store', help='plan file name which has pipeline configurations')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    print(f'Values={values}')


    mode = get_mode()
    version = values.version
    #xclbin = f'{mode}.xilinx_u280_xdma_201920_3{version}/binary_container_1/binary_container_1.xclbin'
    xclbin = 'binary_container_1.xclbin'


    device = pyxrt.device(0)
    xbin = pyxrt.xclbin(xclbin)
    uuid = device.load_xclbin(xbin)
    iplist = xbin.get_ips()
    for ip in iplist:
        print(ip.get_name())

    
    #lutbin = os.path.join(os.path.dirname(xclbin), '../../', 'none_duplicate_long.uvgrid.txt.bin')
    lutbin = 'none_duplicate_long.uvgrid.txt.bin'
    print(f'Using lut binary file {lutbin}')
    lut = np.fromfile(lutbin, dtype=np.uint32)
    print(f'LUT size is {len(lut)}')
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
            
    print(values)

    p.mainbuf.copy_from_device()
    print(p.mainbuf.nparr.shape)
#            self.mainbuf = Buffer((NUREST, NDOUT, NT_OUTBUF, NUVWIDE,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()
#imshow(p.mainbuf.nparr[0,:,:,0,0])
    #show()


    #filehandler = open(values.plan, 'rb')
    #craco_plan = pickle.load(filehandler)
    #print(craco_plan.values)

    p.candidates.copy_from_device()
    print(np.all(p.candidates.nparr == 0))
    p.boxcar_history.copy_from_device()
    print(np.all(p.boxcar_history.nparr == 0))

    p.fdmt_hist_buf.copy_to_device()
    print('inbuf', hex(p.inbuf.buf.address()))
    print('mainbuf', hex(p.mainbuf.buf.address()))
    print('histbuf', hex(p.fdmt_hist_buf.buf.address()))
    print('fdmt_config_buf', hex(p.fdmt_config_buf.buf.address()))




if __name__ == '__main__':
    _main()
