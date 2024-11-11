#!/usr/bin/env python
"""
Runs MPI pipeline for CRACO.

Copyright (C) CSIRO 2022
"""
# MPI IS fussy about being loaded with rc.threads = False - go figure.

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib

import numpy as np
import os
import sys
import logging
from array import array
from craft.craco_plan import PipelinePlan

from craco import cardcap
from craco.cardcap import NCHAN, NFPGA, NSAMP_PER_FRAME
import craco.card_averager
from craco.card_averager import Averager
from craco.cardcapmerger import CcapMerger
from craco.mpiutil import np2array
from craco.vissource import VisSource,open_source, VisBlock
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import numba
import glob
import craft.sigproc as sigproc
from craft.parset import Parset
from craco.search_pipeline_sink import SearchPipelineSink, VisInfoAdapter
from craco.metadatafile import MetadataFile,MetadataDummy
from scipy import constants
from collections import namedtuple
from craft.craco import ant2bl, baseline_iter
from craco import mpiutil
from craft.cmdline import strrange
from craco.timer import Timer
from craco.prep_scan import ScanPrep
from craco.mpi_appinfo import MpiPipelineInfo
from craco.visblock_accumulator import VisblockAccumulatorStruct
from craco.candidate_writer import CandidateWriter
from craco.snoopy_sender import SnoopySender
from craco.mpi_candidate_buffer import MpiCandidateBuffer
from craco.mpi_tracefile import MpiTracefile
from craco.uvfitsfile_sink import UvFitsFileSink

from craco.tracing import tracing
import pickle

    
log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#numprocs = comm.Get_size()

# realtime SCHED_FIFO priorities
BEAM_PRIORITY = 90
RX_PRIORITY = 91

# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]

def set_scheduler(priority:int, policy=None):
    '''
    This breaks the system
        # setting RT priorities makes MPI deadlcok it appears, even with SCHED_RR.
        # SO try just setting nice for now
    '''
    if policy is None:
        policy = os.SCHED_RR
        
    prio_max = os.sched_get_priority_max(policy)
    assert priority <= prio_max, f'Requested priority {priority} greater than max {prio_max} for policy {policy}'
    
    param = os.sched_param(priority)
    pid = 0 # means current process
    affinity = os.sched_getaffinity(pid)
    try:
        #log.info('Setting scheduler policy to %d with priority %d. Affinity is %s. RR interval is %s ms',
        #policy, priority, affinity, os.sched_rr_get_interval(pid)*1e3)
        #os.sched_setscheduler(pid, policy, param)
        #os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(90))
        nicelevel = -19
        old_nicelevel = os.nice(nicelevel)
        log.info('Set nice level from %d to %d', old_nicelevel, nicelevel)

    except PermissionError:
        log.debug('Did not have permission to set scheduler.')

        

REAL_DTYPE = np.float32 # Transpose/averaging type for real types
# Tranaspose/averagign dtype for complex types  can also be a real type, like np.int16 and somethign willl add an extra dimension.
# But fast card_averager doesn't support it for now.
CPLX_DTYPE = np.float32 


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


def get_transpose_dtype(values, real_dtype=REAL_DTYPE, cplx_dtype=CPLX_DTYPE):
    nbeam = values.nbeams
    nt = values.nt
    nant = values.nant_valid
    nc = NCHAN*NFPGA
    vis_fscrunch = values.vis_fscrunch
    vis_tscrunch = values.vis_tscrunch
    npol = 1 # card averager always averagers pol
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, real_dtype, cplx_dtype)
    return dt

## OK KB - YOU NEED TO TEST THE AVERAGER WITH THE INPUT DATA

class DtypeTransposer:
    def __init__(self, info):
        # OK - now this is where it gets tricky and i wish I could refactor it properlty to get what I'm expecting
        self.nbeam = info.nbeams
        self.nrx = info.nrx
        self.dtype = get_transpose_dtype(info)
        self.dtype_complex = get_transpose_dtype(info, cplx_dtype=np.complex64) # the guaranteed compelx version. SearchPipeline needs it
        self.mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.dtype)
        self.msgsize = self.dtype.itemsize
        self.nmsgs = 1
        
        self.comm = info.pipe_info.mpi_app.rx_beam_comm

        numprocs = self.comm.Get_size()
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        t_barrier = MPI.Wtime()
        #comm.Barrier()
        t_start = MPI.Wtime()

        s_msg = [dtx,
                 (np2array(self.tx_counts),
                  np2array(self.tx_displacements)),
                 self.mpi_dtype]
            
        r_msg = [self.drx,
                 (np2array(self.rx_counts),
                  np2array(self.rx_displacements)),
                 self.mpi_dtype]
        
        self.comm.Barrier()
        log.debug('Alltoallv dtype=%s Tx: %s %s %s Rx: %s %s %s', self.dtype,
                  self.tx_counts, self.tx_displacements, dtx.shape,
                  self.rx_counts, self.rx_displacements,self.drx.shape)   
        
        self.comm.Alltoallv(s_msg, r_msg)
        t_end = MPI.Wtime()
        latency = (t_end - t_start)*1e3

        self.last_latency = latency
            
        #if rank == 0:
        #    print(f'RANK0 Barrier wait: {(t_start - t_barrier)*1e3}ms Transpose latency = {latency}ms')

        #print(f'all2all COMPLETE {rank}/{numprocs} latency={latency} ms {t_start} {t_end}')
        return self.drx


class ByteTransposer:
    def __init__(self, info):
        # OK - now this is where it gets tricky and i wish I could refactor it properlty to get what I'm expecting
        self.nbeam = info.nbeams
        self.nrx = info.nrx
        self.dtype_complex = get_transpose_dtype(info, cplx_dtype=np.complex64) # the guaranteed compelx version. SearchPipeline needs it
        self.dtype = get_transpose_dtype(info)
        self.mpi_dtype = MPI.BYTE
        values = info.values
        assert values.transpose_nmsg > 0, 'Invalid transpose nmsg'

        assert self.dtype.itemsize % values.transpose_nmsg == 0, f'Transpose nmsg must divide evenly into {values.transpose_nmsg}. it was {values.tranpose_nmsg}'
                 
        self.msgsize = self.dtype.itemsize // values.transpose_nmsg
        self.displacement = self.dtype.itemsize
        self.nmsgs = values.transpose_nmsg

        self.comm = info.pipe_info.mpi_app.rx_beam_comm
        numprocs = self.comm.Get_size()

        log.info('Transposing %s with type=%s  size=%s NMSG=%s msgsize=%s numprocs=%s', 
                 self.dtype, self.mpi_dtype, self.dtype.itemsize, self.nmsgs, self.msgsize, numprocs)
        
        assert self.nrx + self.nbeam == numprocs, f'Incorrect number of processes in rx_beam_comm. numproc={numprocs} nbeams={self.nbeam} nrx={self.nrx}'
        self.values = info.values
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        nmsgs = self.nmsgs
        t = Timer()
        log.debug('before barrier')
        self.comm.Barrier() # Barrier can take a super long time and slow everything to a halt. dont do it
        t.tick('barrier')
        log.debug('After barrier')

        assert self.nmsgs * self.msgsize == self.dtype.itemsize, 'for now we have to have whole numbers of messages. TODO: Handle final block'
        for imsg in range(nmsgs):
            msgsize = self.msgsize # TODO: Handle final block
            s_msg = [dtx.view(np.byte),
                     (np2array(self.tx_counts*msgsize),
                      np2array(self.tx_displacements*self.displacement + imsg*msgsize)),
                     self.mpi_dtype]
            
            r_msg = [self.drx.view(np.byte),
                     (np2array(self.rx_counts*msgsize),
                      np2array(self.rx_displacements*self.displacement + imsg*msgsize)),
                     self.mpi_dtype]
            #print('SMSG', s_msg[1:])
            #print('RMSG', r_msg[1:])
            self.comm.Alltoallv(s_msg, r_msg)
            
        t.tick('transpose')
        self.last_timer = t
        log.debug('After alltoall')

        return self.drx

class TransposeSender(DtypeTransposer):
    def __init__(self, info):
        super().__init__(info)
        nbeam = self.nbeam
        # Ranks are [RX[0], RX[1], ... RX[N-1], BEAM[0], Beam[1], ... BEAM[NBEAM-1]]
        nrx = self.nrx

        self.tx_counts[-nbeam:] = 1
        self.tx_displacements[-nbeam:] = np.arange(nbeam, dtype=np.int32)
        self.drx = np.zeros((1), dtype=self.dtype) # Dummy for all toall make zero if possible

    def send(self, dtx):
        assert len(dtx) == self.nbeam
        assert dtx.dtype == self.dtype, f'Attempt to send invalid dtype. expected {self.dtype} but got {dtx.dtype}'
        return self.all2all(dtx)

class TransposeReceiver(DtypeTransposer):
    def __init__(self, info):
        super().__init__(info)
        nbeam = self.nbeam
        nrx = self.nrx
        self.rx_counts[:nrx] = 1 # receive same amount from every tx
        self.rx_displacements[:nrx] = np.arange(nrx, dtype=np.int32)
        self.drx = np.zeros(nrx, dtype=self.dtype)
        self.dtx = np.zeros((1), dtype=self.dtype) # dummy buffer for sending TODO: make zero if possible
        self.drx_complex = self.drx.view(dtype=self.dtype_complex) # make a complex view into the same data

    def recv(self):
        return self.all2all(self.dtx)


def proc_rx_get_headers(proc): 
    '''
    Process 1 card per beam
    1 card = 6 FGPAs
    '''
    pipe_info = proc.pipe_info
    set_scheduler(RX_PRIORITY)
    ccap = open_source(pipe_info)
    log.info('opened source')

    # tell all ranks about the headers
    rx_comm = pipe_info.mpi_app.app_comm
    all_hdrs = rx_comm.gather(ccap.fpga_headers, root=0)
    if all_hdrs is None:
        all_hdrs = [] # Just return Empty if we're not root
    
    log.info('Got headers from other RX')
    proc.ccap = ccap
    return all_hdrs

def proc_rx_get_fid0(proc):
    #all_hdrs = world.bcast(all_hdrs, root=0)
    #info = MpiObsInfo(all_hdrs, pipe_info)
    #log.info('got info')
    pipe_info = proc.pipe_info
    info = proc.obs_info
    ccap = proc.ccap

    nbeam = pipe_info.nbeams
    cardidx = pipe_info.cardid
    values = pipe_info.values
    nrx = info.nrx
    nt = info.nt
    nant = info.nant
    nc = NCHAN*NFPGA
    npol_in = info.npol
    assert pipe_info.mpi_app.app_comm.Get_size() == nrx

    dummy_packet = np.zeros((NCHAN*nbeam, ccap.merger.ntpkt_per_frame), dtype=ccap.merger.dtype)
    log.info('made packet %s this is a thing', dummy_packet.shape)
    rsout = os.path.join(pipe_info.values.outdir, f'rescale/b{ccap.block:02d}/c{ccap.card:02d}/') if pipe_info.values.save_rescale else None

    averager = Averager(nbeam, nant, nc, nt, npol_in, values.vis_fscrunch, values.vis_tscrunch, REAL_DTYPE, CPLX_DTYPE, dummy_packet, values.flag_ants, rescale_output_path=rsout)
    log.info('made averager')
    
    transposer = TransposeSender(info)
    log.info('made transposer')

    # construct a typed list for numba - it's a bit of a pain but it needs to be done this way
    # Just need some types of the data so the list can be typed
    
    log.info('Dummy packet shape %s dtype=%s. Averager output shape:%s dtype=%s', dummy_packet.shape, dummy_packet.dtype, averager.output.shape, averager.output.dtype)

    # Do dummy transpose to warm up and make connections
    transposer.send(averager.output)
    log.info('Transpose warmup complete')

    proc.transposer = transposer
    proc.averager = averager
    
    fid0 = ccap.start()
    # send fid0 to all beams
    #fid0 = world.bcast(fid0, root=pipe_info.rx_processor_rank0)
    
    return fid0

def proc_rx_run(proc):

    ccap = proc.ccap
    info = proc.obs_info
    averager = proc.averager
    transposer = proc.transposer
    cardidx = proc.pipe_info.mpi_app.cardid
    
    pktiter = ccap.packet_iter(proc.pipe_info.requested_nframe)

    best_avg_time = 1e6
    
    t_start = MPI.Wtime()
    timer = Timer()
    

    for ibuf, (packets, fids) in enumerate(pktiter):
        timer.tick('read')
        if ibuf == 0:
            log.info('Received ibuf 0')
        
        expected_fid = info.fid_of_block(ibuf) 

        now = MPI.Wtime()
        read_time = now - t_start
        # so we have to put a dummy value in and add a separate flags array
        avg_start = MPI.Wtime()
        test_mode = info.values.test_mode
        check_fids = True
        if check_fids:            
            for pkt, fid in zip(packets, fids):
                #if pkt is not None:
                #    log.debug('Packet %s %s fid %s %s %s pktiter=%s', type(pkt), type(pkt[0]), type(pkt[1]), type(fid), fid, type(pktiter))
                pktfid = None if pkt is None else pkt['frame_id'][0,0]
                assert pkt is None or pktfid == fid, f'FID did not match for ibuf {ibuf}. Expected {fid} but got {pktfid}'
                assert pkt is None or pktfid == expected_fid, f'FID did not match expected ibuf {ibuf} . Expected {expected_fid} but got {pktfid}'
            timer.tick('check fids')

        if test_mode == 'none':
            averaged = averager.accumulate_packets(packets)
            #if ibuf == 0:
                #np.save(f'iblk0_cardid{cardidx:02d}_packets.npz', packets, allow_pickle=True)
                #np.save(f'iblk0_cardid{cardidx:02d}_averaged.npz', averaged, allow_pickle=True)
                #timer.tick('saveaverage')

            timer.tick('average')
        elif test_mode == 'fid':
            fidnos = np.array([0 if pkt is None else fid for pkt, fid in zip(packets, fids)])
            averaged['ics'] = np.repeat(fidnos, 4)[np.newaxis, np.newaxis, :]
            averaged['cas'] = np.repeat(fidnos, 4)[np.newaxis, np.newaxis, :]
            averaged['vis'] = fidnos[0]
            timer.tick('test-fid')
        elif test_mode == 'cardid':
            averaged['ics'].flat = cardidx #np.arange(averaged['ics'].size)
            averaged['cas'].flat = np.arange(averaged['cas'].size)
            averaged['vis'].flat = np.arange(averaged['vis'].size)
            timer.tick('test-cardid')
        else:
            raise ValueError(f'Invalid test mode {test_mode}')

        avg_end = MPI.Wtime()
        avg_time = avg_end - avg_start
        best_avg_time = min(avg_time, best_avg_time)
        #print('RX times', read_time, avg_time, t_start, now, avg_start, avg_end)
        if avg_time*1e3 > 110:
            log.warning('Averaging time for cardidx=%s ibuf=%s was too long: %s ms best=%s', cardidx,ibuf, avg_time*1e3, best_avg_time*1e3)
            

        transposer.send(averaged)
        timer.tick('transpose')
        transpose_end = MPI.Wtime()
        transpose_time = transpose_end - avg_end
        
        if cardidx == 0 and False:
            read_size_bytes = dummy_packet.nbytes*NFPGA
            size_bytes = averaged.size * averaged.itemsize
            transpose_rate_gbps = size_bytes * 8/1e9/transpose_time
            read_rate = read_size_bytes/1e6 / read_time
            log.info('CARD0 ibuf=%d read time %0.1fms rate=%0.1f MB/s. Transpose %s time=%0.1fms rate=%0.1fGbps. Accumulation time=%0.1fms last_nvalid=%d shape=%s dtype=%s, accum size=%s read size %s. Timer: %s',
                     ibuf,
                     read_time*1e3,
                     read_rate,
                     transposer.last_timer,
                     transpose_time*1e3,
                     transpose_rate_gbps,
                     avg_time*1e3,
                     averager.last_nvalid,
                     averaged.shape,
                     averaged.dtype,
                     size_bytes,
                     read_size_bytes,
                     timer)

        t_start = MPI.Wtime()
        if timer.total.perf > 0.120:
            log.warning('RX loop ibuf=%d proctime exceeded 110ms: %s',ibuf,timer)
            
        timer = Timer(args={'ibuf':ibuf+1})
        values = proc.pipe_info.values
        if ibuf == values.num_msgs -1:
            raise ValueError('Stopped')


class FilterbankSink:
    def __init__(self, prefix, vis_source):
        values = vis_source.values
        beamid = vis_source.pipe_info.beamid
        fname = os.path.join(values.outdir, f'{prefix}_b{beamid:02d}.fil')
        pos = vis_source.skycoord
        npol = 1 # Card averager always outputs 1 pol
        hdr = {'nbits':32,
               'nchans':vis_source.nchan,
               'nifs':npol, 
               'src_raj_deg':pos.ra.deg,
               'src_dej_deg':pos.dec.deg,
               'tstart':vis_source.tstart.utc.mjd,
               'tsamp':vis_source.inttime.to(u.second).value,
               'fch1':vis_source.fch1,
               'foff':vis_source.foff,
               'source_name':vis_source.target
        }
        log.info('Creating filterbank %s with header %s', fname, hdr)

        self.fout = sigproc.SigprocFile(fname, 'wb', hdr)

    def write(self, beam_data):
        '''
        Writes data to the filterbank
        data has shape (nrx, nt,  nchan_per_rx) e.g. (36, 128, 24) 
        '''

        assert beam_data.dtype == np.float32, f'WE set nbits=32 for this filterbank, so if it isnt we need to do some work. Type was {beam_data.dtype}'
        dout = np.transpose(beam_data, [1, 0, 2])
#        if rank == 0:
            #print(dout.shape, dout.flatten().shape, dout.size, dout.mean(), dout.std())
        dout.tofile(self.fout.fin)

    def close(self):
        self.fout.fin.close()

def transpose_beam_get_fid0(proc):
    info = proc.obs_info

    transposer = TransposeReceiver(info)
    # Find first frame ID
    log.info('Recieving dummy transpose for warmup')
    transposer.recv()
    proc.transposer = transposer

    return 0

def transpose_beam_run(proc):
    set_scheduler(BEAM_PRIORITY)
    pipe_info = proc.pipe_info
    info = proc.obs_info

    nbeam = pipe_info.nbeams
    values = pipe_info.values
    beamid = pipe_info.beamid
    nrx = info.nrx
    nt = info.nt
    nant = info.nant
    nc = NCHAN*NFPGA
    numprocs = pipe_info.mpi_app.app_size
    assert numprocs ==  nbeam, f'Invalid MPI setup numprocs={numprocs} nrx={nrx} nbeam={nbeam} expected {nrx + nbeam}'

    if pipe_info.mpi_app.app_rank == 0:
        print(f'ObsInfo {info}')

    os.makedirs(values.outdir, exist_ok=True)

    nf = len(info.vis_channel_frequencies)
    nt = 256 # required by pipeline. TODO: Get pipeline NT correctly
    nbl = info.nbl_flagged    
    iblk = 0 
    transposer = proc.transposer

    #cas_filterbank = FilterbankSink('cas',info)
    ics_filterbank = FilterbankSink('ics',info)
    vis_file = UvFitsFileSink(info)
    vis_accum = VisblockAccumulatorStruct(nbl, nf, nt)

    # Make make fake data to get vis_accum to compile. *sig*
    beam_data_complex = transposer.drx_complex # a view into the same data.
    beam_data = transposer.drx
    vis_block_complex = VisBlock(beam_data_complex['vis'], iblk, info, cas=beam_data['cas'], ics=beam_data['ics'])
    vis_accum.compile(vis_block_complex)

    beam_proc_rank = pipe_info.mpi_app.BEAMPROC_RANK
    beam_comm = pipe_info.mpi_app.beam_chain_comm

    # requested block to planner to get moving

    # let the fits sink see some ddata so it can compile
    vis_file.compile(transposer.drx['vis'])

   # warmup send
    beam_comm.Send(vis_accum.mpi_msg, dest=beam_proc_rank)


    vis_accum_send_req = None
    try:
       for iblk in range(pipe_info.requested_nframe):
            t = Timer(args={'iblk':iblk})
            beam_data = transposer.recv()
            if iblk == 0:
                log.info('got block 0')
            beam_data_complex = transposer.drx_complex # a view into the same data.
            t.tick('transposer')
            #cas_filterbank.write(beam_data['cas'])
            #t.tick('cas')
            ics_filterbank.write(beam_data['ics'])
            t.tick('ics')
            vis_block = VisBlock(beam_data['vis'], iblk, info, cas=beam_data['cas'], ics=beam_data['ics'])
            vis_block_complex = VisBlock(beam_data_complex['vis'], iblk, info, cas=beam_data['cas'], ics=beam_data['ics'])
            t.tick('visblock')
            vis_file.write(vis_block) # can't handle complex vis blocks. *groan* -maybe???'
            t.tick('visfile')
            # We have to complete teh send request before writing to the vis_accum
            if vis_accum_send_req is not None:
                req_finished, _ = vis_accum_send_req.test()
                t.tick('Send req test')
                if req_finished:
                    vis_accum.reset()
                    t.tick('vis reset')
                    vis_accum_send_req = None
                else:
                    raise RuntimeError(f'VisAccum isend for blk {iblk-1} not complete. It should be done! finished={req_finished}')

            vis_accum.write(vis_block_complex)
            t.tick('accumulate')
            if vis_accum.is_full:
                # Send asynchronously. It should finish by the time we get the next frame.
                vis_accum_send_req = beam_comm.Isend(vis_accum.mpi_msg, dest=beam_proc_rank)
                t.tick('Isend')

            if beamid == 0 and False:
                log.info('Beam processing time %s. Pipeline processing time: %s', t, pipeline_sink.last_write_timer)

            if t.total.perf > 0.120 and iblk > 0:
                log.warning('Beam loop iblk=%d proctime exceeded 110ms: %s', iblk, t)




    finally:
        print(f'Closing beam files for {beamid}')
        #cas_filterbank.close()
        ics_filterbank.close()
        vis_file.close()
        vis_accum.close()


def proc_beam_run(proc):
    #set_scheduler(BEAM_PRIORITY)
    t = Timer()
    pipe_info = proc.pipe_info
    info = proc.obs_info

    nbeam = pipe_info.nbeams
    values = pipe_info.values
    beamid = pipe_info.beamid
    nrx = info.nrx
    numprocs = pipe_info.mpi_app.app_size
    assert numprocs ==  nbeam, f'Invalid MPI setup numprocs={numprocs} nrx={nrx} nbeam={nbeam} expected {nrx + nbeam}'
    iblk = 0
    update_uv_blocks = pipe_info.values.update_uv_blocks
    # number of bytes in plan message. Otherwise we get an MPI Truncate error. 
    # Pickled an craco_plan and got to 4.1M, so 8M seems OK. 
    # see https://groups.google.com/g/mpi4py/c/AW26x6-fc-A

    PLAN_MSG_SIZE = 16*1024*1024
    planner_rank = pipe_info.mpi_app.PLANNER_RANK # rank of planner - probably should get from mpi_app
    transposer_rank = pipe_info.mpi_app.BEAMTRAN_RANK
    candproc_rank = pipe_info.mpi_app.CANDPROC_RANK
    beam_comm = pipe_info.mpi_app.beam_chain_comm


    # request plan from planner
    planner_iblk = 0 # start at this iblk
    log.info('Requesting iblk %d from planner %d', planner_iblk, planner_rank)
    t.tick('Req plan0')
    beam_comm.send(planner_iblk, dest=planner_rank) # tell planner to make plan starting on this block
    t.tick('req plan0 sent')
    req = beam_comm.irecv(PLAN_MSG_SIZE, source=planner_rank)
    t.tick('irecv plan0')    

    plan_data = req.wait()
    t.tick('plan0 wait')
    plan = plan_data['plan']
    assert plan_data['iblk'] == 0, f'Expected plan for blk=. Was {plan_data}'
    log.info('plan received')

    # now get everything organised
    pipeline_sink = SearchPipelineSink(info, plan)
    t.tick('Make sink')
    nf = len(info.vis_channel_frequencies)
    nt = 256 # required by pipeline. TODO: Get pipeline NT correctly
    nbl = info.nbl_flagged    
    vis_accum = VisblockAccumulatorStruct(nbl, nf, nt)
    t.tick('Make visaccum')
    pipeline_data = vis_accum.pipeline_data
    
    # warumup recv
    beam_comm.Recv(vis_accum.mpi_msg, source=transposer_rank)
    t.tick('warmup recv')
    cand_buf = MpiCandidateBuffer.for_tx(beam_comm, candproc_rank)
    t.tick('make cand buf')

    # Now request next plan while we get organised
    planner_iblk = update_uv_blocks
    beam_comm.send(planner_iblk, dest=planner_rank) # tell planner to make plan starting on this block
    req = beam_comm.irecv(PLAN_MSG_SIZE, source=planner_rank)
    

    try:
        for iblk in range(pipe_info.requested_nsearch):
            t = Timer(args={'iblk':iblk})
            # recieve from transposer rank
            beam_comm.Recv(vis_accum.mpi_msg, source=transposer_rank)

            t.tick('recv')
            if iblk == 0:
                log.info('got block 0')

            candidates = pipeline_sink.write_pipeline_data(pipeline_data, cand_buf.cands)
            t.tick('pipeline')

            # for first block we already have a plan so we don't want to recv a new one straight away
            # Otherwise it all takes too long for the first block, which sux
            if pipeline_sink.ready_for_next_plan:
                plan_received, plan = req.test()
                t.tick('recv plan')

                if plan_received:
                    pipeline_sink.set_next_plan(plan)
                    planner_iblk += update_uv_blocks
                    t.tick('set next plan')
                    beam_comm.send(planner_iblk, dest=planner_rank)
                    req = beam_comm.irecv(PLAN_MSG_SIZE, source=planner_rank)    


            if candidates is None:
                ncand = 0
                #maxsnr = 0
            else:
                ncand = len(candidates)
                #maxsnr = cand_buf.cands['snr'].max()

            cand_buf.send(ncand)
            t.tick('sendcand')

            if beamid == 0 and False:
                log.info('Beam processing time %s. Pipeline processing time: %s', t, pipeline_sink.last_write_timer)

            if t.total.perf > 0.120 and iblk > 0:
                log.warning('Beam loop iblk=%d proctime exceeded 110ms: %s', iblk, t)

            iblk += 1



    finally:
        log.info(f'Closing search pipeline for beam {beamid}')
        beam_comm.send(-1, dest=planner_rank) # Tell planner to quit
        pipeline_sink.close()


class Processor:
    def __init__(self, pipe_info):
        self.trace_file = MpiTracefile.instance()
        self.pipe_info = pipe_info
        
        is_rx = pipe_info.mpi_app.is_rx_processor
        world = self.pipe_info.mpi_app.world
        hdrs = self.get_headers()
        log.debug(f'Headers: Got {len(hdrs)} hdrs before bcast')
        if len(hdrs) > 0:
            log.debug('Headers: %s', hdrs)
        hdrs = world.bcast(hdrs, root=0)
        log.debug(f'Headers: Got {len(hdrs)} hdrs after bcast')
        world.Barrier()
        log.debug('Barrier1')
        self.obs_info = MpiObsInfo(hdrs, pipe_info)
        world.Barrier()
        log.debug('Barrier2')
        fid0 = self.get_fid0()
        world.Barrier()
        log.debug('Barrier3')
        log.debug('FID0 before bcast %d', fid0)
        fid0 = world.bcast(fid0, root=0)
        log.debug('FID0 after bcast %d',fid0)
        self.obs_info.fid0 = fid0            
        log.info(f'fid0={fid0} {type(fid0)}')

    def get_headers(self):
        return ''
    
    def get_fid0(self):
        return 0
    
    def run(self):
        pass

class RxProcessor(Processor):
    def get_headers(self):
        return proc_rx_get_headers(self)
    def get_fid0(self):
        return proc_rx_get_fid0(self)
    def run(self):
        return proc_rx_run(self)

class BeamTransposer(Processor):
    def get_fid0(self):
        # Need to run dummy tranpose before fid0 is broadcast, otehrwise we get
        # a deadlock
        return transpose_beam_get_fid0(self)

    def run(self):
        return transpose_beam_run(self)

class BeamProcessor(Processor):
    def run(self):
        return proc_beam_run(self)

class PlannerProcessor(Processor):
    def run(self):
        count = 0
        app = self.pipe_info.mpi_app
        self.app = app
        comm = self.pipe_info.mpi_app.beam_chain_comm
        self.comm = comm
        fid0 = self.obs_info.fid0        
        log.info(f'fid0={fid0} {type(fid0)}')
        update_uv_blocks = self.pipe_info.values.update_uv_blocks
        iblk = 0
        adapter = VisInfoAdapter(self.obs_info, iblk)
        plan = None # we'll make a plan in the first loop
        # Need to send initial plan to candidate pipeline, but not to beamproc, because beamproc made its own.
        # YUK this smells.
        self.dump_plans = True
        beamid = self.pipe_info.beamid
        self.plan_dout = f'beam{beamid:02d}/plans/'
        os.makedirs(self.plan_dout, exist_ok=True)
        self._candproc_send_req = None

        while True:
            log.info('Waiting for request from %d. Expect iblk=%d', app.BEAMPROC_RANK, iblk)
            t = Timer(args={'iblk':iblk})
            req_iblk = comm.recv(source=app.BEAMPROC_RANK)
            t.tick('recv')
            if req_iblk == -1:
                break
            assert iblk == req_iblk, f'My count of iblk={iblk} but requested was {req_iblk}'        
            adapter = VisInfoAdapter(self.obs_info, iblk)
            t.tick('adapter')
            log.info('Got request to make plan for iblk=%d. My count=%d adapter=%s', req_iblk, iblk, adapter)
            plan = PipelinePlan(adapter, self.obs_info.values, prev_plan=plan)
            t.tick('make plan')
            plan.fdmt_plan # lazy evaluate fdmt_plan
            t.tick('make fdmt plan')
            plan_data = {'count':count, 'iblk': iblk, 'plan':plan}

            # blocking send - it just sends the data. It continues whether or not the beam has received it.
            # To not just fill buffers for ever, we'll need to wait for an ack from the beam
            plan.prev_plan = None # Don't send previous plan - otherwise we maek a nice linked list to all the plans!!!!
            plan.fdmt_plan.prev_pipeline_plan = None # Same for this! Yikes
            plan.fdmt_plan.container1 = None
            plan.fdmt_plan.container2 = None
            comm.send(plan_data, dest=app.BEAMPROC_RANK)            
            self.send_to_candproc(iblk, plan)
            t.tick('candproc')
            log.info('Plan sent count=%d iblk=%d', count, iblk)

            count += 1
            iblk += update_uv_blocks


    def send_to_candproc(self, iblk, plan):
        wcs_data = {'iblk':iblk, 'hdr':plan.fits_header(iblk)}

        if self._candproc_send_req is not None:
            self._candproc_send_req.wait()
            self._candproc_send_req = None


        self._candproc_send_req = self.comm.isend(wcs_data, dest=self.app.CANDPROC_RANK)
        if self.dump_plans:                        
            fout =os.path.join(self.plan_dout, f'plan_iblk{iblk}.pkl')
            with open(fout, 'wb') as f:
                pickle.dump(plan,f)
                sz = os.path.getsize(fout)
                log.info('Wrote %d plan to %s. Size %d', iblk, fout, sz)

class BeamCandProcessor(Processor):
    def run(self):
        t = Timer()
        app = self.pipe_info.mpi_app
        rx_comm = app.beam_chain_comm
        cand_buff = MpiCandidateBuffer.for_rx(rx_comm, app.BEAMPROC_RANK)
        out_cand_buff = MpiCandidateBuffer.for_beam_processor(app.cand_comm)
        from craco.candpipe import candpipe
        candout_dir = 'results/clustering_output'
        os.makedirs(candout_dir, exist_ok=True)
        beamid = self.pipe_info.beamid
        candfname = f'candidates.b{beamid:02d}.txt'
        try:
            os.symlink(os.path.join('../', candfname), os.path.join('results', candfname))
        except FileExistsError:
            pass
                
        candpipe_args = candpipe.get_parser().parse_args(['-o', candout_dir])
        
        pipe = candpipe.Pipeline(self.pipe_info.beamid, 
                                 args=candpipe_args,  
                                 config=None,  # use defaults
                                 src_dir='.', 
                                 anti_alias=True)
        log.info('Made pipeline')
        self.pipe = pipe
        
        trace_file = MpiTracefile.instance()
        # blocking wait for wcs data
        log.info('Waiting for WCS from %d', app.PLANNER_RANK)
        wcs_data = rx_comm.recv(source=app.PLANNER_RANK)
        self.iblk = 0
        log.info('Got first WCS %s', wcs_data)
        self.new_wcs = wcs_data
        self.check_wcs()
        
        # make request for next wcs data
        wcs_req = rx_comm.irecv(source=app.PLANNER_RANK)

        for my_iblk in range(self.pipe_info.requested_nsearch):
            # async receive as we want to async transmit so the pipeeline
            # isn't slowed by the beam cand processor       
            t = Timer(args={'iblk':self.iblk})
            wcs_received, new_wcs = wcs_req.test()
            if wcs_received:
                self.new_wcs = new_wcs
                wcs_req = rx_comm.irecv(source=app.PLANNER_RANK)
                t.tick('RX wcs')
            
            cands = cand_buff.recv()
            ncand = len(cands)
            t.tick('recv')

            maxsnr_in = cands['snr'].max() if ncand > 0 else 0
            log.info('Got %d candidates in BeamCandProc maxsnr=%0.2f', ncand, maxsnr_in)
            #ncand_out = self.single_beam_process(cand_buff.cands[:ncand], out_cand_buff)

            self.check_wcs()
            out_df = pipe.process_block(cands, out_cand_buff.cands)
            maxsnr = out_cand_buff.cands['snr'].max() if len(out_df) > 0 else 0
            log.info('Candidates nin=%d nout=%d maxsn_df=%f maxsn_cands=%f', ncand, len(out_df), out_df['snr'].max(), maxsnr)            
            t.tick('process')

            # gather to everyone - synchronous
            # SEnd MAX_NCAND_OUT per process every time. 
            # Non-existent candiates have -1 as s/n
            
            out_cand_buff.gather()
            t.tick('gather')
            self.iblk += 1

    def set_wcs(self, wcs_data):        
        hdr = wcs_data['hdr']
        iblk = wcs_data['iblk']
        self.pipe.set_current_psf(iblk,hdr)

    def check_wcs(self):
        if self.iblk == self.new_wcs['iblk']:
            self.set_wcs(self.new_wcs)
        

def format_candidate_slack_message(bestcand_dict, outdir):
    '''
    Format a message for the slack channel
    '''
    outdir_split = outdir.split("/")
    sbid = int(outdir_split[4][2:])
    scan = outdir_split[-2]; tstart = outdir_split[-1]

    url = f"""http://localhost:8024/candidate?sbid={sbid}&beam={bestcand_dict["ibeam"]}&scan={scan}&tstart={tstart}&runname=results"""
    url += f"""&dm={bestcand_dict["dm_pccm3"]}&boxcwidth={bestcand_dict["boxc_width"]}&lpix={bestcand_dict["lpix"]}&mpix={bestcand_dict["mpix"]}"""
    url += f"""&totalsample={bestcand_dict["total_sample"]}&ra={bestcand_dict["ra_deg"]}&dec={bestcand_dict["dec_deg"]}"""

    #######################################################################################################
    msg = f'REALTIME CANDIDATE TRIGGERED {bestcand_dict} during scan {outdir}\n Click the link - {url}'

    return msg

class CandMgrProcessor(Processor):
    def run(self):
        app = self.pipe_info.mpi_app
        tx_comm = app.cand_comm
        nbeams = self.obs_info.nbeams
        iblk = 0
        self.cand_writer = CandidateWriter('all_beam_cands.txt', first_tstart=self.obs_info.tstart)
        self.cand_sender = SnoopySender()
        cands = MpiCandidateBuffer.for_beam_manager(app.cand_comm)
        # libpq.so.5 is only installed on root node. This way it only runs on the root node.
        # SHoud make slackpostmanager not reference psycopg2
        from craco.craco_run.auto_sched import SlackPostManager

        self.slack_poster = SlackPostManager(test=False, channel="C05Q11P9GRH")
        for iblk in range(self.pipe_info.requested_nsearch):
            t = Timer(args={'iblk':iblk})
            valid_cands = cands.gather()
            t.tick('Gather')
            if len(valid_cands) > 0:
                self.multi_beam_process(valid_cands)

            t.tick('Multi process')

    def multi_beam_process(self, valid_cands):
        '''
        :cands: CandidateWriter.out_dype np array candidates. Should be len > 1
        '''        
        maxidx = np.argmax(valid_cands['snr'])

        bestcand = valid_cands[maxidx]        
        self.cand_writer.update_latency(valid_cands)
        self.cand_writer.write_cands(valid_cands)
        trace_file = MpiTracefile.instance()
        log.info('Got %d candidates. Best candidate from beam %d was %s', len(valid_cands), bestcand['ibeam'], bestcand)        
        outdir = self.pipe_info.values.outdir
        bestcand_dict = {k:bestcand[k] for k in bestcand.dtype.names}

        if bestcand['snr'] >= self.pipe_info.values.trigger_threshold:
            log.critical('Sending candidate %s', bestcand_dict)
            self.cand_sender.send(bestcand)
            trace_file += tracing.InstantEvent('CandidateTrigger', args=bestcand_dict, ts=None, s='g')
            msg = format_candidate_slack_message(bestcand, outdir)
            self.slack_poster.post_message(msg)

def processor_class_factory(app):                    
    if app.is_beam_transposer:
        proc_klass = BeamTransposer
    elif app.is_beam_processor:
        proc_klass = BeamProcessor
    elif app.is_rx_processor:
        proc_klass = RxProcessor
    elif app.is_planner_processor:
        proc_klass = PlannerProcessor
    elif app.is_cand_processor:
        proc_klass = BeamCandProcessor
    elif app.is_cand_manager:
        proc_klass = CandMgrProcessor
    else:
        raise ValueError(f'Case not handled {app.app_num} {app.rank_info}')
    
    return proc_klass

def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from craco import search_pipeline
    
    cardcap_parser = cardcap.get_parser()
    pipeline_parser = search_pipeline.get_parser()
    parser = ArgumentParser(description='Run MPI pipeline', formatter_class=ArgumentDefaultsHelpFormatter, parents=[cardcap_parser, pipeline_parser], conflict_handler='resolve')
    
    parser.add_argument('--fcm', help='Path to FCM file for antenna positions')
    parser.add_argument('--nfpga-per-rx', type=int, default=6, help='Number of FPGAS received by a single RX process')
    parser.add_argument('--vis-fscrunch', type=int, default=6, help='Amount to frequency average visibilities before transpose')
    parser.add_argument('--vis-tscrunch', type=int, default=1, help='Amount to time average visibilities before transpose')
    parser.add_argument('--ncards-per-host', type=int, default=None, help='Number of cards to process per host, helpful to match previous cardcap')
    parser.add_argument('--cardcap-dir', '-D', help='Local directory (per node?) to load cardcap files from, if relevant. If unspecified, just use files from the positional arguments')
    parser.add_argument('--transpose-nmsg', help='Number of messages to break up transpose into', type=int, default=1)
    parser.add_argument('--search-beams', help='Beams to search. e.g. 0-19', type=strrange, default=[])
    parser.add_argument('--save-uvfits-beams', help='Beams to save UV fits files for. Also requires --metadata and --fcm. e.g. 0-19', type=strrange, default=[])
    parser.add_argument('--dead-cards', help='List of dead cards to avoid. e.g.seren-01:1,seren-04:2', default='')
    parser.add_argument('--save-rescale', action='store_true', default=False, help='Save rescale data to numpy files')
    parser.add_argument('--test-mode', help='Send test data through transpose instead of real data', choices=('fid','cardid','none'), default='none')
    parser.add_argument('--proc-type', help='Process type')
    parser.add_argument('--trigger-threshold', help='Threshold for trigger to send for voltage dump', type=float, default=9.0)
    parser.add_argument(dest='files', nargs='*')
    
    parser.set_defaults(verbose=False)

    return parser

def _main():

    parser = get_parser()
    values = parser.parse_args()
    
    mpiutil.setup_logging(MPI.COMM_WORLD, values.verbose)

    try :        
        pipe_info = MpiPipelineInfo(values)
        
        if values.dump_rankfile:
            sys.exit(0)

        app = pipe_info.mpi_app
        proc_klass = processor_class_factory(app)
        processor = proc_klass(pipe_info)
        processor.run()
    
        log.info(f'run() complete. Waiting for everything')
        raise StopIteration('Throwing a stop error so we can bring down the pipeline')
        #comm.Barrier()
    except:
        log.exception('Exception running pipeline')
        raise # raise it again so the interpreter dies and MPIRUN cleans it up

    
    if rank == 0:
        log.info('Pipeline complete')

#    raise StopIteration('Im raising a stop so it tears down the whole shebang. Perahps a bad idea. Maybe we should look at sending null form the receivers so the beams know theyre done and can shutdown clean')
    

if __name__ == '__main__':
    _main()
