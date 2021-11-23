#!/usr/bin/env python
import numpy as np
from pylab import *
import os
import pyxrt
from craco_testing.pyxrtutil import *
import time
import pickle

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

class PipelinePlan(object):
    def __init__(self, f, values):
        self.values = values
        logging.info('making Plan values=%s', values)

        umax, vmax = f.get_max_uv()
        lres, mres = 1./umax, 1./vmax
        baselines = f.baselines
        nbl = len(baselines)
        freqs = f.channel_frequencies

        # Cant handle inverted bands - this is assumed all over the code. It's boring
        assert freqs.min() == freqs[0]
        assert freqs.max() == freqs[-1]
        Npix = values.npix

        if values.cell is not None:
            lcell, mcell = map(craco.arcsec2rad, values.cell.split(','))
            los, mos = lres/lcell, mres/mcell
        else:
            los, mos = map(float, values.os.split(','))
            lcell = lres/los
            mcell = mres/mos
            
        lfov = lcell*Npix
        mfov = mcell*Npix
        ucell, vcell = 1./lfov, 1./mfov
        fmax = freqs.max()
        foff = freqs[1] - freqs[0]
        lambdamin = 3e8/fmax
        umax_km = umax*lambdamin/1e3
        vmax_km = vmax*lambdamin/1e3
        
        logging.info('Nbl=%d Fch1=%f foff=%f nchan=%d lambdamin=%f uvmax=%s max baseline=%s resolution=%sarcsec uvcell=%s arcsec uvcell= %s lambda FoV=%s deg oversampled=%s',
                 nbl, freqs[0], foff, len(freqs), lambdamin, (umax, vmax), (umax_km, vmax_km), np.degrees([lres, mres])*3600, np.degrees([lcell, mcell])*3600., (ucell, vcell), np.degrees([lfov, mfov]), (los, mos))

        #print(baselines)
        
        uvcells = get_uvcells(baselines, (ucell, vcell), freqs, Npix)
        logging.info('Got Ncells=%d uvcells', len(uvcells))
        d = np.array([(v.a1, v.a2, v.uvpix[0], v.uvpix[1], v.chan_start, v.chan_end) for v in uvcells], dtype=np.int32)
        np.savetxt(values.uv+'.uvgrid.txt', d, fmt='%d',  header='ant1, ant2, u(pix), v(pix), chan1, chan2')

        self.uvcells = uvcells
        self.nd = values.ndm
        self.nt = values.nt
        self.freqs = freqs
        self.npix = Npix
        self.nbox = values.nbox
        self.boxcar_weight = values.boxcar_weight
        self.nuvwide = values.nuvwide
        self.nuvmax = values.nuvmax
        assert self.nuvmax % self.nuvwide == 0
        self.nuvrest = self.nuvmax // self.nuvwide
        self.ncin = values.ncin
        self.ndout = values.ndout
        self.foff = foff
        self.dtype = np.complex64 # working data type
        self.threshold = values.threshold
        self.nbl = nbl
        self.fdmt_scale = self.values.fdmt_scale
        self.fft_scale = self.values.fft_scale
        self.fft_ssr = 16 # number of FFT pixels per clock - "super sample rate"
        self.ngridreg = 16 # number of grid registers to do
        assert self.threshold >= 0, 'Invalid threshold'
        self.fdmt_plan = FdmtPlan(uvcells, self)
        self.save_fdmt_plan_lut()

        
        if self.fdmt_plan.nuvtotal >= values.nuvmax:
            raise ValueError("Too many UVCELLS")

        self.upper_instructions = calc_grid_luts(self, True)
        self.lower_instructions = calc_grid_luts(self, False)
        self.save_grid_instructions(self.upper_instructions, 'upper')
        self.save_grid_instructions(self.lower_instructions, 'lower')
        self.upper_idxs, self.upper_shifts, self.lower_idxs, self.lower_shifts = calc_pad_lut(self, self.fft_ssr)
        self.save_pad_lut(self.upper_idxs, self.upper_shifts, 'upper')
        self.save_pad_lut(self.lower_idxs, self.lower_shifts, 'lower')

        filehandler = open("pipeline.obj", 'wb') 
        pickle.dump(self, filehandler)
        filehandler.close()
        
    def save_lut(self, data, lutname, header, fmt='%d'):
        filename = '{uvfile}.{lutname}.txt'.format(uvfile=self.values.uv, lutname=lutname)
        logging.info('Saving {lutname} shape={d.shape} type={d.dtype} to {filename} header={header}'.format(lutname=lutname, d=data, filename=filename, header=header))
        np.savetxt(filename, data, fmt=fmt, header=header)

    def save_fdmt_plan_lut(self):
        fruns = self.fdmt_plan.runs
        d = []
        for irun, run in enumerate(fruns):
            for icell, cell in enumerate(run.cells):
                d.append([cell.a1,
                          cell.a2,
                          cell.uvpix[0],
                          cell.uvpix[1],
                          cell.chan_start,
                          cell.chan_end,
                          irun,
                          icell,
                          run.total_overlap,
                          run.max_idm,
                          run.max_offset,
                          run.offset_cff,
                          run.idm_cff,
                          run.fch1])

        d = np.array(d)

        header='ant1, ant2, u(pix), v(pix), chan1, chan2, irun, icell, total_overlap, max_idm, max_offset, offset_cff, idm_cff, fch1'
        fmt = '%d ' * 8 + ' %d '*3 + ' %f '*3
        self.save_lut(d, 'uvgrid.split', header, fmt=fmt)
        

        
    def save_grid_instructions(self, instructions, name):
        logging.info('Got %d %s grid instructions', len(instructions), name)
        d = np.array([[i.target_slot, i.uvidx, i.shift_flag, i.uvpix[0], i.uvpix[1]] for i in instructions], dtype=np.int32)
        header ='target_slot, uvidx, shift_flag, upix, vpix'
        self.save_lut(d, 'gridlut.'+name, header)

    def save_pad_lut(self, idxs, shifts, name):
        d = np.array(idxs, dtype=np.int32)
        header ='upix, vpix, regidx'
        self.save_lut(d, 'padlut.'+name, header)

        d = np.array(shifts, dtype=np.int32)
        header = 'doshift'
        self.save_lut(d, 'doshift.'+name, header)
        


    @property
    def nf(self):
        '''Returns number of frequency channels'''
        return len(self.freqs)


    @property
    def fmin(self):
        '''
        Returns maximum frequency
        '''
        return self.freqs[0]

    @property
    def fmax(self):
        '''
        Returns minimum frequency
        '''
        return self.freqs[-1]

    @property
    def dmax(self):
        '''
        Returns maximum DM - placeholder for when we do DM gaps
        '''
        return self.nd
      
class FdmtRun(object):
    def __init__(self, cells, plan):
        self.plan = plan
        mincell = min(cells, key=lambda cell:cell.chan_start)
        self.cells = cells
        self.chan_start = mincell.chan_start
        self.fch1 = mincell.fch1
        self.total_overlap = 0
        for uv in cells:
            overlap = calc_overlap(uv, self.chan_start, plan.pipeline_plan.ncin)
            logging.debug('Cell chan_start %s %s %s-%s overlap=%d', self.chan_start, uv, uv.chan_start, uv.chan_end, overlap)
            assert overlap > 0
            self.total_overlap += overlap

        assert self.max_idm <= plan.pipeline_plan.ndout, 'NDOUT is too small - needs to be at least %s' % self.max_idm

    @property
    def offset_cff(self):
        return craco_kernels.offset_cff(self.fch1, self.plan.pipeline_plan)

    @property
    def idm_cff(self):
        return craco_kernels.idm_cff(self.fch1, self.plan.pipeline_plan)

    @property
    def max_idm(self):
        dmax = self.plan.pipeline_plan.dmax
        return int(np.ceil(dmax*self.idm_cff))

    @property
    def max_offset(self):
        dmax = self.plan.pipeline_plan.dmax
        return int(np.ceil(dmax*self.offset_cff))

    def __str__(self):
        ncells = len(self.cells)
        return 'ncells={ncells} fch1={self.fch1} chan_start={self.chan_start} total_overlap={self.total_overlap}'.format(self=self, ncells=ncells)
        

class FdmtPlan(object):
    def __init__(self, uvcells, pipeline_plan):
        self.pipeline_plan = pipeline_plan
        nuvwide = self.pipeline_plan.nuvwide
        ncin = self.pipeline_plan.ncin
        uvcells_remaining = uvcells[:] # copy array
        fdmt_runs = []
        run_chan_starts = []
        run_fch1 = []
        runs = []
        while len(uvcells_remaining) > 0:
            logging.debug('Got %d/%d uvcells remaining', len(uvcells_remaining), len(uvcells))
            minchan = min(uvcells_remaining, key=lambda uv:(uv.chan_start, uv.blid)).chan_start
            possible_cells = filter(lambda uv:calc_overlap(uv, minchan, ncin) > 0, uvcells_remaining)

            # Do not know how to get a length of iterator in python3, comment it out here
            #logging.debug('Got %d possible cells', len(possible_cells))

            # sort as best we can so that it's stable - I.e. we get hte same answer every time
            best_cells = sorted(possible_cells, key=lambda uv:(calc_overlap(uv, minchan, ncin), uv.blid, uv.upper_idx), reverse=True)
            logging.debug('Got %d best cells. Best=%s overlap=%s', len(best_cells), best_cells[0], calc_overlap(best_cells[0], minchan, ncin))
            used_cells = best_cells[0:min(nuvwide, len(best_cells))]
            full_cells, leftover_cells = split_cells(used_cells, minchan, ncin)
            run = FdmtRun(full_cells, self)
            run_chan_starts.append(run.chan_start)
            run_fch1.append(run.fch1)
            fdmt_runs.append(full_cells)
            runs.append(run)
            # create lookup table for each run
            
            total_overlap = run.total_overlap
            # Do not know how to get a length of iterator in python3, comment it out here
            #logging.debug('minchan=%d npossible=%d used=%d full=%d leftover=%d total_overlap=%d', minchan, len(possible_cells), len(used_cells), len(full_cells), len(leftover_cells), total_overlap)
            
            # Remove used cells
            uvcells_remaining = [cell for cell in uvcells_remaining if cell not in used_cells]

            # Add split cells
            uvcells_remaining.extend(leftover_cells)
            
        nruns = len(fdmt_runs)
        nuvtotal = nruns*nuvwide

        ndout = self.pipeline_plan.ndout
        nd = self.pipeline_plan.nd
        nt = self.pipeline_plan.nt
        #square_history_size = ndout*nuvtotal*(nt + nd)
        square_history_size = sum(nuvwide*(nd + nt)*ndout for run in runs)
        minimal_history_size = sum(nuvwide*(run.max_offset+ nt)*run.max_idm for run in runs)
        efficiency = float(len(uvcells))/float(nuvtotal)
        required_efficiency = float(nuvtotal)/8192.0
        
        logging.info('FDMT plan has ntotal=%d of %d runs with packing efficiency %f. Grid read requires efficiency of > %f of NUV=8192. History size square=%d minimal=%d =%d 256MB HBM banks', nuvtotal, nruns, efficiency, required_efficiency, square_history_size, minimal_history_size, minimal_history_size*4/256/1024/1024)
        self.fdmt_runs = fdmt_runs
        self.run_chan_starts = run_chan_starts
        self.run_fch1 = run_fch1
        self.runs = runs

        # create an FDMT object for each run so  we can use it to calculate the lookup tbales
        #     def __init__(self, f_min, f_off, n_f, max_dt, n_t, history_dtype=None):

        fdmts = [fdmt.Fdmt(fch1, self.pipeline_plan.foff, ncin, ndout, 1) for fch1 in self.run_fch1]
        fdmt_luts = np.array([thefdmt.calc_lookup_table() for thefdmt in fdmts])
        niter = int(np.log2(ncin))
        # final LUTs we need to copy teh same LUT for every NUVWIDE
        assert fdmt_luts.shape == (nruns, ncin-1, 2)
        self.fdmt_lut = np.repeat(fdmt_luts[:,:,np.newaxis, :], nuvwide, axis=2)
        expected_lut_shape = (nruns, ncin-1, nuvwide, 2)
        assert self.fdmt_lut.shape == expected_lut_shape, 'Incorrect shape for LUT=%s expected %s' % (self.fdmt_lut.shape, expected_lut_shape)
        

        self.nruns = nruns
        self.nuvtotal = nuvtotal
        self.total_nuvcells = sum([len(p) for p in fdmt_runs])

        # find a cell with zero in it
        self.zero_cell = None
        for irun, run in enumerate(self.runs):
            if len(run.cells) < nuvwide:
                self.zero_cell = (irun, len(run.cells))

        if self.zero_cell is None:
            self.zero_cell = (len(self.runs), 0)

        assert self.zero_cell != None
        assert self.zero_cell[0] < self.pipeline_plan.nuvrest, 'Not enough room for FDMT zero cell'
        assert self.zero_cell[1] < nuvwide

        assert self.zero_cell[0] < self.nruns
        assert self.zero_cell[1] < ncin
        #assert self.get_cell(self.zero_cell) != None

        logging.info("FDMT zero cell is %s=%s", self.zero_cell, self.zero_cell[0]*nuvwide+self.zero_cell[1])

        uvmap = {}
        for irun, run in enumerate(self.runs):
            for icell, cell in enumerate(run.cells):
                irc = (irun, icell)
                uvmap[cell.uvpix_upper] = uvmap.get(cell.uvpix_upper, [])
                uvmap[cell.uvpix_upper].append(irc)

        self.__uvmap = uvmap

    def cell_iter(self):
        '''
        Iteration over all the cells
        '''
        for run in self.runs:
            for cell in run.cells:
                yield cell

    def find_uv(self, uvpix):
        '''
        Returns the run and cell index of FDMT Cells that have the given UV pixel

        uvpix must be upper hermetian
        
        Returns a list of tuples
        irun = run index
        icell = cellindex inside the run

        You can find the cell with 
        self.get_cell((irun, icell))
        
        :uvpix: 2-tuple of (u, v)
        :returns: 2-typle (irun, icell)
        '''

        assert uvpix[0] >= uvpix[1], 'Uvpix must be upper hermetian'
        # speed optimisation
        cell_coords2 = self.__uvmap.get(uvpix,[])
        
        #cell_coords = []
        #for irun, run in enumerate(self.runs):
        #    for icell, cell in enumerate(run.cells):
        #        if cell.uvpix_upper == uvpix:
        #            cell_coords.append((irun, icell))


        #print(uvpix, 'Version1', cell_coords, 'Verion2', cell_coords2)
        #assert cell_coords == cell_coords2

        return cell_coords2

    def get_cell(self, cell_coord):
        irun, icell = cell_coord
        if cell_coord == self.zero_cell:
            return None
        else:
            return self.runs[irun].cells[icell]

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
        
        
class Pipeline:
    def __init__(self, device, xbin, lut):
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
    parser.add_argument('-p', '--pickle', default='pipeline.obj', type=str, action='store', help='pickle file name which has pipeline configurations')
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
    p = Pipeline(device, xbin, lut)
    
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


    filehandler = open(values.pickle, 'rb')
    craco_plan = pickle.load(filehandler)
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
