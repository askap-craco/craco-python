#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging


from scipy import constants
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

from craco.cardcap import NCHAN, NFPGA, NSAMP_PER_FRAME
from craco.cardcapmerger import CcapMerger
from craco.prep_scan import ScanPrep
from craco.metadatafile import MetadataFile,MetadataDummy
from craft.craco import ant2bl, baseline_iter


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"



class MpiObsInfo:
    '''
    Gets headers from everyone and tells everyone what they need to know
    uses lots of other classes in a hacky way to interpret the headers, and merge of the headers
    '''
    def __init__(self, hdrs, pipe_info):
        '''
        '''
        self.pipe_info = pipe_info
        values = pipe_info.values
        self.values = values
        # make megers for all the receiers
        # just assume beams will have returned ['']
        # I'm not sure we need card mergers any more
        #self.card_mergers = []
        #for card_hdrs in hdrs:
        #    if len(card_hdrs) > 0 and len(card_hdrs[0]) > 0:
        #        self.card_mergers.append(CcapMerger.from_headers(card_hdrs))
                
        all_hdrs = []

        # flatten into all headers
        for card_hdrs in hdrs:
            for fpga_hdr in card_hdrs:
                if len(fpga_hdr) != 0:
                    all_hdrs.append(fpga_hdr)

        self.main_merger = CcapMerger.from_headers(all_hdrs)
        m = self.main_merger
        self.raw_freq_config = m.freq_config
        self.vis_freq_config = self.raw_freq_config.fscrunch(self.values.vis_fscrunch)

        self.__fid0 = None # this gets sent with some transposes later

        outdir = pipe_info.values.outdir
        indir = outdir.replace('/data/craco/','/CRACO/DATA_00/') # yuck, yuck, yuck - but I don't have time for this right now
        log.info('Loading scan prep from %s', indir)
        self._prep = ScanPrep.load(indir)
        
        if values.metadata is not None:
            self.md = MetadataFile(values.metadata)
            valid_ants_0based = np.arange(self.nant)
        else:
            if self.pipe_info.mpi_app.is_in_beam_chain: # beamid only defined for beam processors
                self.md = self._prep.calc_meta_file(self.beamid)
            else:
                self.md = None

            valid_ants_0based = np.array(self._prep.valid_ant_numbers) - 1
            #self.md = MetadataDummy()

        assert np.all(valid_ants_0based >= 0)
        flag_ants_0based = set(np.array(self.values.flag_ants) - 1)
        # Make sure we remove antenans 31-36 (1 based) = 30-35 (00based)
        self.valid_ants_0based = np.array(sorted(list(set(valid_ants_0based) - set(flag_ants_0based) - set(np.arange(6) + 30))))
        log.info('Valid ants: %s', self.valid_ants_0based+1)
        #assert len(self.valid_ants_0based) == self.nant - len(self.values.flag_ants), 'Invalid antenna accounting'
  
    def sources(self):
        '''
        Returns an ordered dict
        '''
        assert self.md is not None, 'Requested source list but not metadata specified'
        srcs =  self.md.sources(self.beamid)
        assert len(srcs) > 0

        return srcs

    def source_index_at_time(self, mjd:Time):
        '''
        Return the 0 based index in the list of sources at the specified time
        '''
        assert self.md is not None, 'Requested source index but no metadata specified'
        s = self.md.source_index_at_time(mjd)
        return s

    def uvw_at_time(self, mjd:Time):
        '''
        Returns np array (nant, 3) UVW values in seconds at the given time
        '''
        uvw = self.md.uvw_at_time(mjd, self.beamid)[self.valid_ants_0based, :]  /constants.c #convert to seconds

        return uvw

    def baselines_at_time(self, mjd:Time):
        return self.md.baselines_at_time(mjd, self.valid_ants_0based, self.beamid)

    def antflags_at_time(self, mjd:Time):
        '''
        Return antenna flags at given time
        :see:MetadadtaFile for format
        '''
        flags =  self.md.flags_at_time(mjd)[self.valid_ants_0based]
        return flags

    def baseline_iter(self):
        return baseline_iter(self.valid_ants_0based)
                
    @property
    def xrt_device_id(self):
        '''
        Returns which device should be used for searcing for this pipeline
        Returns None if this beam processor shouldnt be processing this beam
        '''
        
        beam_rank_info = self.pipe_info.beam_rank_info(self.beamid)
        xrtdev = beam_rank_info.xrt_device_id
        
        return xrtdev

    @property
    def beamid(self):
        return self.pipe_info.beamid

    @property
    def nbeams(self):
        '''
        Number of beams being processed
        '''
        return self.pipe_info.nbeams

    @property
    def nt(self):
        '''
        Number of samples per block
        '''
        return self.pipe_info.nt

    @property
    def nrx(self):
        '''
        Number of receiver processes
        '''
        return self.pipe_info.nrx
    
    @property
    def target(self):
        '''
        Target name
        '''
        return self.main_merger.target

    def fid_to_mjd(self, fid:int)->Time:
        '''
        Convert the given frame ID into an MJD
        '''
        return self.main_merger.fid_to_mjd(fid)

    @property
    def vis_fscrunch(self):
        '''
        Requested visibility fscrunch factor
        '''
        return self.values.vis_fscrunch

    @property
    def vis_nt(self) ->int:
        '''
        visiblity NT per block after tscrunch in teh transpose dtype
        '''
        vnt  = self.nt // self.vis_tscrunch
        return vnt
    
    @property
    def vis_nc(self) -> int:
        '''
        Visibility NChan after fscrunch in the transpose dtype
        '''
        vnc = NCHAN*NFPGA // self.vis_fscrunch
        return vnc

    @property
    def vis_tscrunch(self):
        '''
        Requested visibility tscrunch factor
        '''
        return self.values.vis_tscrunch
        
    @property
    def fid0(self):
        '''
        Frame ID of first data value
        '''
        return self.__fid0

    @fid0.setter
    def fid0(self, fid):
        '''
        Set frame iD of first data value
        '''
        self.__fid0 = np.uint64(fid)
        tnow = Time.now()
        tstart = self.tstart
        diff = (tstart.tai - tnow.tai)
        diffsec = diff.to(u.second)
        log.info('Set FID0=%d. Tstart=%s = %s = %0.1f seconds from now', self.__fid0, tstart, tstart.iso, diffsec.value)

    @property
    def tstart(self):
        '''
        Returns astropy of fid0
        We assume we start on the frame after fid0
        '''
        assert self.__fid0 is not None, 'First frame ID must be set before we can calculate tstart'
        fid_first = self.fid_of_block(0)
        return self.main_merger.ccap[0].time_of_frame_id(fid_first)

    def fid_of_block(self, iblk):
        '''
        Returns the frame ID at the beginning? of the BEAMFORMER block index indexed by idex
        idx=0 = first ever beamformer block we receive
        '''
        return self.fid0 + np.uint64(iblk)*np.uint64(NSAMP_PER_FRAME)

    def fid_of_search_block(self, siblk):
        '''
        Returns frame ID at the beginning of a SEARCH block index.
        Note that a search block is always 256 samples, where as a BEAMFORMER block / frame is always 110ms.
        Handling all this is a headache
        '''
        assert siblk >= 0
        nint_per_search_block = 256
        nint_per_frame = self.main_merger.nint_per_frame
        nsamp_per_int = self.vis_tscrunch * np.uint64(NSAMP_PER_FRAME) // nint_per_frame  # number of FIDS per integration. Including tscrunch    
        nsamp_per_search_block = nint_per_search_block * nsamp_per_int
        fid = self.fid0 + np.uint64(siblk)*np.uint64(nsamp_per_search_block)

        return fid

    @property
    def nchan(self):
        '''
        Number of channels of input before fscrunch
        '''
        return self.raw_freq_config.nchan

    @property
    def npol(self):
        '''
        Number of polarisations of input source
        '''
        return self.main_merger.npol

    @property
    def fch1(self):
        '''
        First channel frequency
        '''
        return self.raw_freq_config.fch1

    @property
    def fcent(self):
        '''
        Central frequency of band
        '''
        return self.raw_freq_config.fcent

    @property
    def foff(self):
        '''
        Offset Hz between channels
        '''
        return self.raw_freq_config.foff

    @property
    def vis_channel_frequencies(self):
        '''
        Channel frequencies in Hz of channels in the visibilities after 
        f scrunching
        '''
        return self.vis_freq_config.channel_frequencies

    @property
    def skycoord(self) -> SkyCoord:
        '''
        Astorpy skycoord of given beam
        '''
        # For the metadata file we calculate the start time and use that
        # to work out which source is being used,
        # then load the source table and get the skycoord
        tstart = self.tstart
        if self.md is None: # return a default onne in case metadata not set - just for filterbanks
            coord = SkyCoord('00h00m00s +00d00m00s' )
        else:
            source = self.md.source_at_time(self.pipe_info.beamid, tstart)
            coord = source['skycoord']
            
        return coord

    @property
    def inttime(self):
        '''
        Returns instegration time as quantity in seconds
        '''
        return self.main_merger.inttime*u.second

    @property
    def vis_inttime(self):
        '''
        Returns quantity in seconds
        '''
        return self.inttime*self.values.vis_tscrunch # do I need to scale by vis_tscrunch? Main merger already has it?

    @property
    def nt(self):
        '''
        Return number of integrations per beamformer frame
        '''
        return self.main_merger.nt_per_frame

    @property
    def nant(self):
        '''
        Returns the number of antennas in teh source data.
        Not necessarily the number of antennas that are valid or that will be transposed
        '''
        #log.info('Nant is %d nant_valid is %d valid ants: %s', self.main_merger.nant, self.nant_valid, self.valid_ants_0based)
        return self.main_merger.nant

    @property
    def nant_valid(self):
        '''
        Returns the number of antennas after flagging
        '''
        return len(self.valid_ants_0based)

    @property
    def nbl_flagged(self):
        '''
        Returns number of baselines after flagging
        '''
        na = self.nant_valid
        nb = na * (na -1) // 2
        return nb

    def __str__(self):
        m = self.main_merger
        s = f'fch1={m.fch1} foff={m.foff} nchan={m.nchan} nant={m.nant} inttime={m.inttime}'
        return s


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
