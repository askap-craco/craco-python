#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.ccapfits2uvfits import get_antennas
from craft.corruvfits import CorrUvFitsFile
import scipy
from craft.parset import Parset
from craco.timer import Timer
from craco.cardcap import NCHAN, NFPGA, NSAMP_PER_FRAME
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from craco.vissource import VisBlock
from numba import njit

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def prepare_data_slow(vis_block, total_nchan, npol, baselines):
    t = Timer()
    vis_data = vis_block.data
    
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
    fid_start = vis_block.fid_start
    fid_mid = vis_block.fid_mid
    mjd = vis_block.mjd_mid
    antflags = vis_block.antflags
    samps_per_vis = np.uint64(NSAMP_PER_FRAME // vis_nt)
    sourceidx = vis_block.source_index
    uvw = vis_block.uvw
    t.tick('prep')
    if np.iscomplexobj(vis_data):
        dreshape = np.transpose(vis_data, (3,1,0,2)).reshape(vis_nt, nbl, total_nchan, npol) # should be [t, baseline, coarsechan*finechan]
        t.tick('transpose')
        damp = abs(dreshape)
        t.tick('amp')
    else:
        # transpose takes 30 ms
        dreshape = np.transpose(vis_data, (3,1,0,2,4)).reshape(vis_nt, nbl, total_nchan, npol,2) # should be [t, baseline, coarsechan*finechan]
        t.tick('transpose')

        # amplitude takes 14 ms
        damp = np.sqrt(dreshape[...,0]**2 + dreshape[...,1]**2)
        t.tick('amp')


    log.debug('Input data shape %s, output data shape %s', vis_data.shape, dreshape.shape)

    blflags = vis_block.baseline_flags

    weights = np.ones((vis_nt, nbl, total_nchan, npol), dtype=np.float32)
    weights[damp == 0] = 0 # flag channels that have zero amplitude
    uvw_baselines = np.empty((nbl, 3))

    # fits convention has source index with starting value of 1
    t.tick('prep weights')

    for blinfo in baselines:
        ia1 = blinfo.ia1
        ia2 = blinfo.ia2
        a1 = blinfo.a1
        a2 = blinfo.a2
        ibl = blinfo.blidx
        uvw_baselines[ibl, :] = uvw[ia1, :] - uvw[ia2, :]
        if blflags[ibl]:
            weights[:, ibl, ...] = 0
            dreshape[:, ibl,...] = 0 # set output to zeros too, just so we can't cheat

    t.tick('apply weights')

    #log.debug('UVFITS block %s fid_start=%s fid_mid=%s info.nt=%s vis_nt=%s fid_itime=%s mjd=%s=%s inttime=%s', self.blockno, fid_start, fid_mid, info.nt, vis_nt, fid_itime, mjd, mjd.iso, inttime)
    #        self.uvout.put_data_block(uvw_baselines, mjd.value, self.blids, inttime, dreshape[itime, ...], weights[itime, ...], fits_sourceidx)

    return (dreshape, weights, uvw_baselines)  

def write_data_slow(uvout, uvw_baselines,dreshape, weights, fits_sourceidx, mjds, blids, inttime):
    # UV Fits files really like being in time order
    vis_nt, nbl, total_nchan, npol = weights.shape

    for itime, mjd in enumerate(mjds):

        #log.debug('UVFITSfid_start=%s fid_mid=%s info.nt=%s vis_nt=%s fid_itime=%s mjd=%s=%s inttime=%s', fid_start, fid_mid, info.nt, vis_nt, fid_itime, mjd, mjd.iso, inttime)
        uvout.put_data_block(uvw_baselines, mjd, blids, inttime, dreshape[itime, ...], weights[itime, ...], fits_sourceidx)

@njit(cache=True) # damn - njit doesn't support big endian on intel.
def prep_data_fast_numba_tscrunch(dout, vis_data, uvw_baselines, iblk, inttim):
    '''
    dout is the dtype = np.dtype([('UU', dt), ('VV', dt), ('WW', dt), \
            ('DATE', dt), ('BASELINE', dt), \
            ('FREQSEL', dt), ('SOURCE', dt), ('INTTIM', dt), \
            ('DATA', dt, (1, 1, 1, nchan, npol, self.ncomplex))])

    it has shape (vis_nt_out, nbl)

    ncomplex = 2 if no flags, and 3 if there are flags

    vis_data is the input and is [nrx, nbl, vis_nc, vis_nt ]

    inttim is the integration time in days per input sample
    output data can be an integer fraction less than the input, in which case it does tscrunching
    '''
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]

    vis_nt_out, nbl_out = dout.shape
    assert vis_nt % vis_nt_out == 0
    assert nbl_out == nbl

    tscrunch = vis_nt // vis_nt_out
    scale = np.float32(1./tscrunch)

    vscrunch = np.zeros((2), dtype=np.float32)

    for irx in range(nrx):
        for ibl in range(nbl):
            for ic in range(vis_nc):
                cout = ic + vis_nc*irx
                for it in range(vis_nt_out):
                    isamp = (it + iblk)                    
                    mjddiff = isamp*inttim
                    d = dout[it, ibl]
                    d['UU'], d['VV'], d['WW']= uvw_baselines[ibl,:]
                    d['DATE'] = mjddiff
                    data = d['DATA']
                    vscrunch[:] = 0
                    vstart = vis_data[irx, ibl, ic, it*tscrunch:(it+1)*tscrunch, :]
                    for ix in range(tscrunch):
                        vscrunch[0] += vstart[ix, 0]
                        vscrunch[1] += vstart[ix, 1]

                    vscrunch *= scale
                    
                    if vscrunch[0] == 0 and vscrunch[1] == 0:
                        weight = 0
                    else:
                        weight = 1
                    data[0,0,0,cout,0,:] = [vscrunch[0], vscrunch[1], weight]

@njit(cache=True) # damn - njit doesn't support big endian on intel.
def prep_data_fast_numba(dout, vis_data, uvw_baselines, iblk, inttim):
    '''
    dout is the dtype = np.dtype([('UU', dt), ('VV', dt), ('WW', dt), \
            ('DATE', dt), ('BASELINE', dt), \
            ('FREQSEL', dt), ('SOURCE', dt), ('INTTIM', dt), \
            ('DATA', dt, (1, 1, 1, nchan, npol, self.ncomplex))])

    it has shape (vis_nt_out, nbl)

    ncomplex = 2 if no flags, and 3 if there are flags

    vis_data is the input and is [nrx, nbl, vis_nc, vis_nt ]
    '''
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
    for irx in range(nrx):
        for ibl in range(nbl):
            for ic in range(vis_nc):
                for it in range(vis_nt):
                    cout = ic + vis_nc*irx
                    isamp = it + iblk
                    mjddiff = isamp*inttim                
                    d = dout[it, ibl]
                    d['UU'], d['VV'], d['WW']= uvw_baselines[ibl,:]
                    d['DATE'] = mjddiff
                    data = d['DATA']
                    vis = vis_data[irx,ibl,ic,it,:]
                    if vis[0] == 0 and vis[1] == 0:
                        weight = 0
                    else:
                        weight = 1
                    data[0,0,0,cout,0,:] = [vis[0], vis[1], weight]

def prep_data_fast_numpy(dout, vis_data, uvw_baselines, mjds):
    '''
    dout is the dtype = np.dtype([('UU', dt), ('VV', dt), ('WW', dt), \
            ('DATE', dt), ('BASELINE', dt), \
            ('FREQSEL', dt), ('SOURCE', dt), ('INTTIM', dt), \
            ('DATA', dt, (1, 1, 1, nchan, npol, self.ncomplex))])

    it has length(vis_nt, nbl)

    ncomplex = 2 if no flags, and 3 if there are flags

    vis_data is the input and is [nrx, nbl, vis_nc, vis_nt ]
    '''
    nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
    dout['UU'] = uvw_baselines[:,0][None,:]
    dout['VV'] = uvw_baselines[:,1][None,:]
    dout['WW'] = uvw_baselines[:,2][None,:]
    dout['DATE'] = mjds[:,None]    
    data = dout['DATA']   
    
    for irx in range(nrx):
        for ibl in range(nbl):
            for ic in range(vis_nc):

                cout = ic + vis_nc*irx                
                vis = vis_data[irx,ibl,ic,it,:]
                if vis[0] == 0 and vis[1] == 0:
                    weight = 0
                else:
                    weight = 1
                data[:,:,0,cout:cout+vis_nc,0,:] = vis_data[irx,:,]

    


class DataPrepper:    
    def __init__(self, uvfitsout:CorrUvFitsFile, baselines, vis_nt:int, fits_sourceidx:int, inttim:float, tscrunch:int=1):
        self.uvfitsout = uvfitsout
        dtbe = uvfitsout.dtype # big endian datatype

        dt = np.dtype('<f4')# must be little endian for numba to work
        assert dt.byteorder == '=', 'Internal byte order must be native for numba to work'
        
        self.dtype_le = np.dtype([('UU', dt), 
                                   ('VV', dt), 
                                   ('WW', dt), 
                                    ('DATE', dt),  
                                    ('BASELINE', dt),                                     
                                    ('FREQSEL', dt),
                                    ('SOURCE', dt), 
                                    ('INTTIM', dt), 
                                    ('DATA', dt, dtbe['DATA'].shape)])

        assert vis_nt % tscrunch == 0, f'Invalid tscrunch={tscrunch} for vis_nt={vis_nt}'

        self.baselines = baselines
        self.vis_nt = vis_nt
        self.tscrunch = tscrunch
        self.out_vis_nt = self.vis_nt // self.tscrunch
        self.fits_sourceidx = fits_sourceidx
        self.inttim = inttim
        nbl = len(self.baselines)
        self.uvw_baselines = np.empty((nbl, 3))
        self.dout = np.zeros((self.out_vis_nt, nbl), dtype=self.dtype_le)
        dout = self.dout
        blids = [bl.blid for bl in baselines]
        self.blids =np.array(blids)

        # set the things that are constant
        # Used to be able to do this but you can't because we'r edoing inplace byteswap
        # and you can't byteswap to a different buffe
                ##dout['FREQSEL'] = 1
        #dout['SOURCE'] = fits_sourceidx
        #dout['BASELINE'] = np.array(blids)[None,:]
        #dout['INTTIM'] = inttim

        self.iblk = 0
        self.inttime_days = self.inttim / 86400

    def write(self, vis_block:VisBlock, use_uvws=True):
        #t = Timer()
        vis_data = vis_block.data
        vis_nt = vis_data.shape[3]
        assert self.vis_nt == vis_nt
        if use_uvws:
            uvw = vis_block.uvw
            for blinfo in self.baselines:
                ia1 = blinfo.ia1
                ia2 = blinfo.ia2
                ibl = blinfo.blidx
                self.uvw_baselines[ibl, :] = uvw[ia1, :] - uvw[ia2, :]
        else:
            self.uvw_baselines[:] = 0
        #t.tick('calc baselines')

        dout = self.dout
        # Bulk set these values. IT's eaiser and probably not to slow. 
        # coul dpass into the numba function but might not help much
        # we need to set them again because we did inplace bytswap, which ruins everything
        # Dammit - why is UVFITS so dumb?
        dout['FREQSEL'] = 1
        dout['SOURCE'] = self.fits_sourceidx
        dout['BASELINE'] = self.blids[None,:]
        dout['INTTIM'] = self.inttim*self.tscrunch
        #t.tick('Set constants')

        if self.tscrunch == 1:
            prep_data_fast_numba(self.dout, vis_data, self.uvw_baselines, self.iblk, self.inttime_days)
        else:
            prep_data_fast_numba_tscrunch(self.dout, vis_data, self.uvw_baselines, self.iblk, self.inttime_days*self.tscrunch)

        #t.tick('prep fast')
        v = self.dout.view(np.float32)
        #t.tick('view')
        v.byteswap(inplace=True) # FITS is big endian. Damn.
        #t.tick('byteswap')
        v.tofile(self.uvfitsout.fout)
        #t.tick('tofile')
        self.uvfitsout.ngroups += self.dout.size
        self.iblk += self.out_vis_nt
        return self.dout

    def compile(self, vis_data):
        '''
        Run the numba funciton to compile it
        '''
        prep_data_fast_numba(self.dout, vis_data, self.uvw_baselines, self.iblk, self.inttime_days )

            

class UvFitsFileSink:
    def __init__(self, obs_info, fileout=None, extra_header=None, format='fits', use_uvws=True, fcm=None):

        '''
        Raw does some kidnof raw format that we'll have to use the header to covert

        '''
        assert format in ('fits', 'raw'), f'Invalid format {format}'
        self.format = format
        self.use_uvws = use_uvws
        beamno = obs_info.beamid
        self.beamno = beamno
        self.obs_info = obs_info
        self.blockno = 0
        values = obs_info.values

        
        if fileout is None:
            fileout = os.path.join(values.outdir, f'b{beamno:02}.uvfits')
            
        self.fileout = fileout
        if fcm is None:
            fcm = Parset.from_file(values.fcm)
            
        antennas = get_antennas(fcm)
        log.info('FCM %s contained %d antennas', values.fcm, len(antennas))
        info = obs_info
        fcent = info.fcent
        foff = info.foff * info.vis_fscrunch
        assert info.nchan % info.vis_fscrunch == 0, f'Fscrunch needs to divide nchan {info.nchan} {values.vis_fscrunch}'
        nchan = info.nchan // info.vis_fscrunch
        self.npol = 1 # card averager always sums polarisations
        npol = self.npol
        tstart = (info.tstart.utc.value + info.inttime.to(u.day).value)
        self.total_nchan = nchan
        self.source_list = obs_info.sources().values()
        source_list = self.source_list
        log.info('UVFits sink opening file %s fcent=%s foff=%s nchan=%s npol=%s tstart=%s sources=%s nant=%d', fileout, fcent, foff, nchan, npol, tstart, source_list, len(antennas))
        
        _extra_header = {'BEAMID': beamno, 'TSCALE':'UTC', 'FORMAT':self.format}
        if extra_header is not None:
            _extra_header.update(extra_header)
            
        self.uvout = CorrUvFitsFile(fileout,
                                    fcent,
                                    foff,
                                    nchan,
                                    npol,
                                    tstart,
                                    source_list,
                                    antennas,
                                    extra_header=_extra_header,
                                    instrume='CRACO')

        # create extra tables so we can fix it later on. if the file is not closed properly
        self.uvout.fq_table().writeto(fileout+".fq_table", overwrite=True)
        self.uvout.an_table(self.uvout.antennas).writeto(fileout+'.an_table', overwrite=True)
        self.uvout.su_table(self.uvout.sources).writeto(fileout+'.su_table', overwrite=True)
        self.uvout.hdr.totextfile(fileout+'.header', overwrite=True)
        self.blids = [bl.blid for bl in self.obs_info.baseline_iter()]
        baseline_info = list(self.obs_info.baseline_iter())
        fits_sourceidx = 1
        inttime = self.obs_info.inttime.to(u.second).value*info.vis_tscrunch
        vis_nt = self.obs_info.vis_nt
        self.prepper = DataPrepper(self.uvout, 
                                   baseline_info,
                                   vis_nt, 
                                   fits_sourceidx, 
                                   inttime)


        with open(fileout+'.groupsize', 'w') as fout:
            fout.write(str(self.uvout.dtype.itemsize) + '\n')
            

    def compile(self, vis_data):
        if self.prepper is not None:
            self.prepper.compile(vis_data)

    def write(self, vis_block):
        '''
        vis_data has len(nrx) and shape inner shape
        vishape = (nbl, vis_nc, vis_nt, 2) if np.int16 or
        or 
        vishape = (nbl, vis_nc, vis_nt) if np.complex64
        '''
        if self.uvout is None:
            return

        t = Timer()
        do_fast = True
        if self.format == 'fits':
            if do_fast:
                self.prepper.write(vis_block, self.use_uvws)
                t.tick('Write prepper')
            else:
                info = self.obs_info
                vis_data = vis_block.data
                nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
                fid_start = vis_block.fid_start
                fid_mid = vis_block.fid_mid
                mjd = vis_block.mjd_mid
                antflags = vis_block.antflags
                samps_per_vis = np.uint64(NSAMP_PER_FRAME // vis_nt)
                sourceidx = vis_block.source_index
                fits_sourceidx = sourceidx + 1
                nant = info.nant
                inttime = info.inttime.to(u.second).value*info.vis_tscrunch

                assert nbl == info.nbl_flagged, f'Expected nbl={info.nbl_flagged} but got {nbl}'
                # FID is for the beginning of the block.
                # we might vis_nt = 2 and the FITS convention is to use the integraton midpoint
                fid_itimes = [fid_start + samps_per_vis // 2 + itime*samps_per_vis for itime in range(vis_nt)]
                mjds = np.array([info.fid_to_mjd(fid_itime).utc.value for fid_itime in fid_itimes])
                
                assert NSAMP_PER_FRAME % vis_nt == 0
                (dreshape, weights, uvw_baselines) = prepare_data_slow(vis_block, self.total_nchan, self.npol, info.baseline_iter())
                t.tick('prepare')               
                write_data_slow(self.uvout, uvw_baselines,dreshape, weights, fits_sourceidx, mjds, self.blids, inttime)
                t.tick('Write')
                #self.uvout.fout.flush()
                t.tick('flush')

        elif self.format == 'raw':
            self.write_raw(vis_block)
        else:
            assert self.format in ('fits', 'raw'), f'Invalid format {self.format}'
        
        #if self.beamno == 0:
        #    log.debug(f'File size is {os.path.getsize(self.fileout)} blockno={self.blockno} ngroups={self.uvout.ngroups} timer={t}')
        self.blockno += 1

    def write_raw(self, vis_block:VisBlock):
        t = Timer(args={'iblk':vis_block.iblk})
        if self.use_uvws:
            vis_block.uvw.tofile(self.uvout.fout)
            t.tick('write uvw')
            
        vis_block.data.tofile(self.uvout.fout)
        t.tick('write vis')


    def close(self):
        print(f'Closing file {self.uvout}')
        if self.uvout is not None:
            self.uvout.close()

    def __del__(self):
        self.close()
        

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
