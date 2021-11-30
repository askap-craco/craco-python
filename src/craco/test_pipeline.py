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

from craft import uvfits
from craft import craco

'''
Most hard-coded numebrs are updated
'''

NBLK = 11
NCU = 4
NTIME_PARALLEL = (NCU*2)

NDM_MAX = 1024
HBM_SIZE = int(256*1024*1024)
NBINARY_POINT = 6

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
    def __init__(self, device, xbin, plan):
        self.plan = plan

        # If we are on new version of pipeline
        self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, lut = get_grid_lut_from_plan(self.plan)
        print(f'{self.nparallel_uvin} {self.nparallel_uvout} {self.h_nparallel_uvout}')
        print(f'{lut.shape}')
        
        np.savetxt("lut.txt", lut, fmt="%d")
                
        self.grid_reader = DdgridCu(device, xbin)
        self.grids = [GridCu(device, xbin, i) for i in range(4)]
        self.ffts = [FfftCu(device, xbin, i) for i in range(4)]
        self.boxcarcu = BoxcarCu(device, xbin)
        self.fdmtcu = FdmtCu(device, xbin)

        print(f'lut.shape {lut.shape}')
        print(f'nuv {self.plan.fdmt_plan.nuvtotal}')
        
        print('Allocating grid LUTs')
        # For grid with new version pipeline
        self.grid_luts = [Buffer(lut.shape, np.uint16, device, g.krnl.group_id(5)).clear() for g in self.grids]
        
        for l in self.grid_luts:
            l.nparr[:] = lut
            l.copy_to_device()
                
        # FDMT: (pin, pout, histin, histout, pconfig, out_tbkl)
        print('Allocating FDMT Input')

        # Used to be like this
        #self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.ncin, self.plan.nt, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()
        print(self.plan.fdmt_plan.nuvtotal)
        print(self.plan.nt)
        print(self.plan.ncin)
        print(self.plan.nuvwide)
        print(self.plan.fdmt_plan.nuvtotal*self.plan.nt*self.plan.ncin*self.plan.nuvwide)        
        self.inbuf = Buffer((self.plan.fdmt_plan.nuvtotal, self.plan.nt, self.plan.ncin, self.plan.nuvwide, 2), np.int16, device, self.fdmtcu.krnl.group_id(0)).clear()        
        
        # FDMT histin, histhout should be same buffer
        assert self.fdmtcu.group_id(2) == self.fdmtcu.group_id(3), 'FDMT histin and histout should be the same'
        
        print('Allocating FDMT history')
        # Use a whole HBM for history FDMT
        self.fdmt_hist_buf = Buffer((HBM_SIZE), np.int8, device, self.fdmtcu.krnl.group_id(2), 'device_only').clear() # Grr, group_id puts you in some weird addrss space self.fdmtcu.krnl.group_id(2))
        
        #print('Allocating FDMT fdmt_config_buf')
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
        #self.mainbuf = Buffer((self.plan.nuvrest, self.plan.ndout, nt_outbuf, self.plan.nuvwide,2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()
        self.mainbuf = Buffer((self.plan.nuvrest, self.plan.ndout, NBLK, self.plan.nt, self.plan.nuvwide, 2), np.int16, device, self.grid_reader.krnl.group_id(0)).clear()

        print('Allocating ddreader_lut')
        self.ddreader_lut = Buffer((NDM_MAX + self.plan.nuvrest), np.uint32, device, self.grid_reader.group_id(5)).clear()
        print('Allocating boxcar_history')    

        npix = self.plan.npix
        self.boxcar_history = Buffer((self.plan.nd, self.plan.nbox - 1, npix, npix), np.int16, device, self.boxcarcu.group_id(3), 'device_only').clear() # Grr, gruop_id problem self.boxcarcu.group_id(3))
        print('Allocating candidates')

        candidate_dtype=np.dtype([('snr', np.uint16), ('loc_2dfft', np.uint16), ('boxc_width', np.uint8), ('time', np.uint8), ('dm', np.uint16)])

        # The buffer size here should match the one declared in C code
        self.candidates = Buffer(NDM_MAX*self.plan.nbox, candidate_dtype, device, self.boxcarcu.group_id(5)).clear() # Grrr self.boxcarcu.group_id(3))


def run(p, blk, values):
    self = p

    # To do it properly we need to get number from plan
    threshold = self.plan.threshold
    threshold = np.uint16(threshold*(1<<NBINARY_POINT))
    ndm       = self.plan.nd

    nchunk_time = self.plan.nt//NTIME_PARALLEL
    nuv         = self.plan.fdmt_plan.nuvtotal

    tblk = blk % NBLK
    
    nparallel_uv = nuv//2
    nurest       = nuv//8
    
    load_luts = 1

    nplane = ndm*nchunk_time
    shift1 = 0 # FFT CONFIG register - not sure what this means
    shift2 = 7 # FFT CONFIG Register - not sure what this means
    fft_cfg = (nplane << 16) + (shift2 << 6) + (shift1 << 3)

    print(f'\nConfiguration just before pipeline running \nndm={ndm} nchunk_time={nchunk_time} tblk={tblk} nuv={nuv} nparallel_uv={nparallel_uv} nurest={nurest} load_luts={load_luts} nplane={nplane} threshold={threshold} shift1={shift1} shift2={shift2} fft_cfg={fft_cfg}\n')

    assert ndm < 1024 # It hangs for 1024 - not sure why.

    starts = []
    
    assert nparallel_uv == self.nparallel_uvin # the number from pipeline plan should be the same as we calculated based on indexs from pipeline plan
        
    if values.run_image:
        for cu in self.ffts:
            starts.append(cu(fft_cfg, fft_cfg))
            
        starts.append(self.boxcarcu(ndm, nchunk_time, threshold, self.boxcar_history, self.boxcar_history, self.candidates))
        starts.append(self.grid_reader(self.mainbuf, ndm, tblk, nchunk_time, nurest, self.ddreader_lut, load_luts))

        for cu, grid_lut in zip(self.grids, self.grid_luts):
            starts.append(cu(ndm, nchunk_time, self.nparallel_uvin, self.nparallel_uvout, self.h_nparallel_uvout, grid_lut, load_luts))
            
    if values.run_fdmt:
        starts.append(self.fdmtcu(self.inbuf, self.mainbuf, self.fdmt_hist_buf, self.fdmt_hist_buf, self.fdmt_config_buf, nurest, tblk))

    return starts


def location2pix(location, npix):

    npix_half = npix//2
    
    vpix = (location//npix)%npix - npix_half
    if (vpix<0):
        vpix = npix+vpix
        
    upix = location%npix - npix_half
    if (upix<0):
        upix = npix+upix
        
    #location_index = ((npix_half+vpix)%npix)*npix + (npix_half+upix)%npix
    return vpix, upix

def print_candidates(candidates, npix):
    print(f"snr\t(vpix, upix)\tboxc_width\ttime\tdm")
    #for candidate in np.sort(candidates):
    for candidate in candidates:
        location = candidate['loc_2dfft']
        vpix, upix = location2pix(location, npix)

        snr = float(candidate['snr'])/float(1<<NBINARY_POINT) 
        print(f"{snr:.3f}\t({upix}, {vpix})\t{candidate['boxc_width']+1}\t\t{candidate['time']}\t{candidate['dm']}")
    
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose',   action='store_true', help='Be verbose')
    parser.add_argument('-r', '--run_fdmt',  action='store_true', help='Run FDMT pipeline')
    parser.add_argument('-R', '--run_image', action='store_true', help='Run Image pipeline')
    parser.add_argument('-w', '--wait',      action='store_true', help='Wait during execution')
    
    parser.add_argument('-b', '--nblocks',   action='store', type=int, help='Number of blocks')
    parser.add_argument('-d', '--device',    action='store', type=int, help='Device number')
    parser.add_argument('-n', '--npix',      action='store', type=int, help='Number of pixels in image')
    parser.add_argument('-c', '--cell',      action='store', type=int, help='Image cell size (arcsec). Overrides --os')
    parser.add_argument('-m', '--ndm',       action='store', type=int, help='Number of DM trials')
    parser.add_argument('-t', '--nt',        action='store', type=int, help='Number of times per block')
    parser.add_argument('-B', '--nbox',      action='store', type=int, help='Number of boxcar trials')
    parser.add_argument('-U', '--nuvwide',   action='store', type=int, help='Number of UV processed in parallel')
    parser.add_argument('-N', '--nuvmax',    action='store', type=int, help='Maximum number of UV allowed.')
    parser.add_argument('-C', '--ncin',      action='store', type=int, help='Numer of channels for sub fdmt')
    parser.add_argument('-D', '--ndout',     action='store', type=int, help='Number of DM for sub fdmt')
    
    parser.add_argument('-T', '--threshold', action='store', type=float, help='Threshold for candidate grouper')
    
    parser.add_argument('-o', '--os',        action='store', type=str, help='Number of pixels per beam')
    
    parser.add_argument('-x', '--xclbin',    action='store', type=str, help='XCLBIN to load.')
    parser.add_argument('-u', '--uv',        action='store', type=str, help='Load antenna UVW coordinates from this UV file')
    parser.add_argument('-s', '--show',      action='store_true',      help='Show plots')
    
    # These three are not used in PipelinePlan ...
    parser.add_argument('-W', '--boxcar_weight', type=str,   help='Boxcar weighting type', choices=('sum','avg','sqrt'), default='sum')
    parser.add_argument('-f', '--fdmt_scale',    type=float, help='Scale FDMT output by this amount')
    parser.add_argument('-F', '--fft_scale',     type=float, help='Scale FFT output by this amount. If both scales are 1, the output equals the value of frb_amp for crauvfrbsim.py')
    
    parser.set_defaults(verbose   = False)
    parser.set_defaults(run_fdmt  = False)
    parser.set_defaults(run_image = False)
    parser.set_defaults(wait      = False)
    parser.set_defaults(show      = False)
    
    parser.set_defaults(nblocks   = 1)
    parser.set_defaults(device    = 0)
    parser.set_defaults(npix      = 256)
    parser.set_defaults(ndm       = 2)
    parser.set_defaults(nt        = 256)
    parser.set_defaults(nbox      = 8)
    parser.set_defaults(nuvwide   = 8)
    parser.set_defaults(nuvmax    = 8192)
    parser.set_defaults(ncin      = 32)
    parser.set_defaults(ndout     = 186) # used to be 32
    parser.set_defaults(threshold = 3.0)
    parser.set_defaults(boxcar_weight = "sum")
    parser.set_defaults(fdmt_scale =1.0)
    parser.set_defaults(fft_scale  =10.0)
    
    parser.set_defaults(os        = "2.1,2.1")
    parser.set_defaults(xclbin    = "binary_container_1.xclbin.golden")
    parser.set_defaults(uv        = "frb_d0_lm0_nt16_nant24.fits")
    
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

    # Create a plan
    f = uvfits.open(values.uv)
    plan = PipelinePlan(f, values)

    # Create a pipeline 
    p = Pipeline(device, xbin, plan)

    fast_baseline2uv = craco.FastBaseline2Uv(plan)
    uv_shape     = (plan.nuvrest, plan.nt, plan.ncin, plan.nuvwide)
    uv_out  = np.zeros(uv_shape, dtype=np.complex64)
    
    #duration = end - start
    #print(f'fast version of Baseline2uv took {duration} seconds')
    #
    #return dout


    # inbuf is the input to FDMT
    #p.inbuf.nparr[:][0] = 1
    #p.inbuf.nparr[:][1] = 0
    #p.inbuf.copy_to_device()

    # mainbuf is the input to pipeline
    #(self.plan.nuvrest, self.plan.ndout, nt_outbuf, self.plan.nuvwide, 2)    
    p.mainbuf.nparr[:,:,:,:,0] = 1
    p.mainbuf.nparr[:,:,:,:,1] = 0
    p.mainbuf.copy_to_device()

    if values.wait:
        input('Press any key to continue...')
        
    for iblk, input_data in enumerate(f.time_blocks(plan.nt)):
        if iblk >= values.nblocks:
            break

        input_flat = craco.bl2array(input_data)
        fast_baseline2uv(input_flat, uv_out)

        # mainbuf shape is different from uv_out shape
        print(uv_out.shape)
        print(p.mainbuf.nparr.shape)
        
        #p.mainbuf.nparr[:,:,:,:,:] = uv_out

        #p.mainbuf.nparr = uv_out.astype(np.int16) # not sure how to do this
        p.mainbuf.copy_to_device()
        
        # Now we need to use baselines data    
        call_start = time.perf_counter()
        starts = run(p, iblk, values)
        wait_start = time.perf_counter()
    
        for istart, start in enumerate(starts):
            print(f'Waiting for istart={istart} start={start}')
            start.wait(0)

            wait_end = time.perf_counter()
            print(f'Call: {wait_start - call_start} Wait:{wait_end - wait_start}: Total:{wait_end - call_start}')

    f.hdulist.close()
    
    p.mainbuf.copy_from_device()
    print(p.mainbuf.nparr.shape)

    p.candidates.copy_from_device()
    p.boxcar_history.copy_from_device()
    print(np.all(p.boxcar_history.nparr == 0))

    p.fdmt_hist_buf.copy_to_device()
    print('inbuf', hex(p.inbuf.buf.address()))
    print('mainbuf', hex(p.mainbuf.buf.address()))
    print('histbuf', hex(p.fdmt_hist_buf.buf.address()))
    print('fdmt_config_buf', hex(p.fdmt_config_buf.buf.address()))

    # Copy data from device
    p.candidates.copy_from_device()
    candidates = p.candidates.nparr[:]

    # Find first zero output
    last_candidate_index = np.where(candidates['snr'] == 0)[0][0]
    candidates = candidates[0:last_candidate_index]

    print_candidates(candidates, p.plan.npix)
                     
if __name__ == '__main__':
    _main()
