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
from craco import cardcap
from craco.cardcap import NCHAN, NFPGA, NSAMP_PER_FRAME
import craco.card_averager
from craco.card_averager import Averager
from craco.cardcapmerger import CcapMerger
from craco.mpiutil import np2array
from craco.vissource import VisSource,open_source
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import numba
import glob
import craft.sigproc as sigproc
from craft.parset import Parset
from craco.search_pipeline_sink import SearchPipelineSink
from craco.metadatafile import MetadataFile
from scipy import constants
from collections import namedtuple
from craft.craco import ant2bl
from craco import mpiutil
from craft.cmdline import strrange

    
log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]


class BaselineIndex:
    def __init__(self, blidx, a1, a2, ia1, ia2):
        '''
        blidx - index into an array that's had flagged antenans removed
        a1 - 1 based antenna number
        a2 - 1 based antenna number
        ia1 = index into an array of antennas that's had the flagged antennas removed
        ia2 = index into an array of antennas that's had the flagged antennas removed
        '''
        assert a1 > 0 and a2 > 0
        self.blidx = blidx
        self.a1 = a1
        self.a2 = a2
        self.ia1 = ia1
        self.ia2 = ia2

    @property
    def blid(self):
        '''
        Returns the baseline ID (FITS formatted) of this index
        '''
        return ant2bl((self.a1, self.a2))

    def __str__(self):
        s = f'BaselineIndex blidx={self.blidx} {self.a1}-{self.a2} idx={self.ia1}-{self.ia2} blid={self.blid}'
        return s

    __repr__ = __str__
        

REAL_DTYPE = np.float32 # Transpose/averaging type for real types
CPLX_DTYPE = np.complex64 # Tranaspose/averagign dtype for complex types  can also be a real type, like np.int16 and somethign willl add an extra dimension

class BeamRankInfo(namedtuple('BeamProcInfo', ['beamid','rank','host','slot','core','xrt_device_id'])):
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Beam {self.beamid} xrtdevid={self.xrt_device_id}'
        return s

class ReceiverRankInfo(namedtuple('ReceiverInfo', ['rxid','rank','host','slot','core','block','card','fpga'])):
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Block {self.block} card {self.card} fpga {self.fpga}'
        return s


class MpiPipelineInfo:
    def __init__(self, values):
        '''
        TODO - integrate this in with a nice way of doing dump hostfile
        '''
        self.hosts = mpiutil.parse_hostfile(values.hostfile)
        self.beam_ranks = []
        self.receiver_ranks = []

        if values.max_ncards is None:
            ncards = len(values.block)*len(values.card)
        else:
            ncards = values.max_ncards

        nrx = ncards*len(values.fpga)//values.nfpga_per_rx
        self.nrx = nrx
        self.ncards = ncards

        # yuck. This is just yuk.
        if values.beam is None:
            values.nbeams = 36
        else:
            assert 0<= values.beam < 36, f'Invalid beam {values.beam}'
            values.nbeams = 1

        
        self.nbeams = values.nbeams
        self.world_comm = MPI.COMM_WORLD
        self.world_rank = comm.Get_rank()
        self.values = values
        color = 1 if self.is_rx_processor else 0
        # a communicator that splits by rx and everything else
        self.rx_comm = comm.Split(color)

    @property
    def is_beam_processor(self):
        ''' 
        First nbeams ranks are beam processors
        Send nrx ranks are RX processor
        '''
        bproc = self.world_rank < self.nbeams
        return bproc

    @property
    def rx_processor_rank0(self):
        return self.nbeams # Rank0 for rx processor is

    @property
    def beamid(self):
        assert self.is_beam_processor, 'Requested beam ID of non beam processor'
        return self.world_rank

    def beam_rank_info(self, beamid):
        '''
        Returns the rank info for the given beam
        '''
        
        rank_info = next(filter(lambda info: info.beamid == beamid, self.beam_ranks))
        return rank_info

    @property
    def is_rx_processor(self):
        return not self.is_beam_processor

    @property
    def cardid(self):
        values = self.values
        assert self.is_rx_processor, 'Requested cardID of non-card processor'
        theid = self.world_rank - self.nbeams
        return theid


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
        self.card_mergers = []
        for card_hdrs in hdrs:
            if len(card_hdrs) > 0 and len(card_hdrs[0]) > 0:
                self.card_mergers.append(CcapMerger.from_headers(card_hdrs))
                
        #self.card_mergers = [CcapMerger.from_headers(card_hdrs)
        #                     for card_hdrs in hdrs if len(card_hdrs[0]) != 0]
        
        all_hdrs = []

        # flatten into all headers
        for card_hdrs in hdrs:
            for fpga_hdr in card_hdrs:
                if len(fpga_hdr) != 0:
                    all_hdrs.append(fpga_hdr)

        self.main_merger = CcapMerger.from_headers(all_hdrs)

        m = self.main_merger
        assert m.fch1 == m.all_freqs.min(), f'Channel 0 = {m.fch1} but lowest channel is {m.all_freqs.min()}'

        # we're goign to assume the individual card receiver mergers and preprocessing
        # will do the right thing. So we're just going to check channels at a card level

        card_freqs = np.array([m.fch1 for m in self.card_mergers])
        if len(self.card_mergers) > 1:
            card_freqdiff = card_freqs[1:] - card_freqs[:-1]
            card_foff = np.abs(card_freqdiff[0] - card_freqdiff)
            assert np.all(card_foff < 1e-3), f'Cards frequencies not contiguous {card_foff} {card_freqs} {card_freqdiff}'
            assert card_freqdiff[0] > 0, f'Card frequency increment should probably be positive. It was {card_freqdiff[0]}'
            
        self.__fid0 = None # this gets sent with some transposes later

        if self.pipe_info.is_beam_processor and values.metadata is not None:
            # TODO: implement calc11 and getting sources from .... elsewhere
            self.md = MetadataFile(values.metadata)
        else:
            self.md = None

        self.valid_ants_0based = np.array([ia for ia in range(self.nant) if ia+1 not in self.values.flag_ants])
        assert len(self.valid_ants_0based) == self.nant - len(self.values.flag_ants), 'Invalid antenna accounting'
  
    def sources(self):
        '''
        Returns an ordered dict
        '''
        assert self.md is not None, 'Requested source list but not metadata specified'
        return self.md.sources(self.beamid)

    def source_index_at_time(self, mjd:Time):
        '''
        Return the 0 based index in the list of sources at the specified time
        '''
        assert self.md is not None, 'Requested source index but no metadata specified'
        s = self.md.source_index_at_time(mjd.value)
        return s

    def uvw_at_time(self, mjd:Time):
        '''
        Returns np array (nant, 3) UVW values in seconds at the given time
        '''
        uvw = self.md.uvw_at_time(mjd.value)[self.valid_ants_0based, self.beamid, :]  /constants.c #convert to seconds
        return uvw

    def antflags_at_time(self, mjd:Time):
        '''
        Return antenna flags at given time
        :see:MetadadtaFile for format
        '''
        flags =  self.md.flags_at_time(mjd.value)[self.valid_ants_0based]
        return flags

    def baseline_iter(self):
        '''
        Returns an iterator over the valid baselines
        Returns a BaselineIndex whichis info on which baselines are present
        No autocorrelations are returned
        '''
        blidx = 0
        for ia1, a1 in enumerate(self.valid_ants_0based):
            for a2 in self.valid_ants_0based[ia1+1:]:
                ia2 = list(self.valid_ants_0based).index(a2)
                b = BaselineIndex(blidx, a1+1, a2+1, ia1, ia2)
                yield b
                blidx += 1

                
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

    def fid_to_mjd(self, fid:int):
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

    @property
    def tstart(self):
        '''
        Returns astropy of fid0
        We assume we start on the frame after fid0
        '''
        assert self.__fid0 is not None, 'First frame ID must be set before we can calculate tstart'
        fid_first = self.fid_of_block(1) # first block is for rescaling 
        return self.main_merger.ccap[0].time_of_frame_id(fid_first)

    def fid_of_block(self, iblk):
        return self.fid0 + np.uint64(iblk + 1)*np.uint64(NSAMP_PER_FRAME)

    @property
    def nchan(self):
        '''
        Number of channels of input before fscrunch
        '''
        return self.main_merger.nchan

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
        return self.main_merger.fch1

    @property
    def fcent(self):
        '''
        Central frequency of band
        '''
        return self.main_merger.fcent

    @property
    def foff(self):
        '''
        Offset Hz between channels
        '''
        return self.main_merger.foff

    @property
    def vis_channel_frequencies(self):
        '''
        Channel frequencies in Hz of channels in the visibilities after 
        f scrunching
        '''
        main_freqs = np.arange(self.nchan)*self.foff + self.fch1
        vis_freqs = main_freqs.reshape(-1, self.vis_fscrunch).mean(axis=1)
        return vis_freqs

    @property
    def skycoord(self) -> SkyCoord:
        '''
        Astorpy skycoord of given beam
        '''
        # For the metadata file we calculate the start time and use that
        # to work out which source is being used,
        # then load the source table and get the skycoord
        tstart = self.tstart
        source = self.md.source_at_time(self.pipe_info.beamid, tstart.mjd)
        coord = source['skycoord']
        return coord

    @property
    def inttime(self):
        return self.main_merger.inttime*u.second

    @property
    def vis_inttime(self):
        return self.inttime # do I need to scale by vis_tscrunch? Main merger already has it?

    @property
    def nt(self):
        '''
        Return number of samples per frame
        '''
        return self.main_merger.nt_per_frame

    @property
    def nant(self):
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


def get_transpose_dtype(values):
    nbeam = values.nbeams
    nt = values.nt
    nant = values.nant_valid
    nc = NCHAN*NFPGA
    vis_fscrunch = values.vis_fscrunch
    vis_tscrunch = values.vis_tscrunch
    npol = 1 # card averager always averagers pol
    rdtype = np.float32
    cdtyep = np.complex64
    dt = craco.card_averager.get_averaged_dtype(nbeam, nant, nc, nt, npol, vis_fscrunch, vis_tscrunch, REAL_DTYPE, CPLX_DTYPE)
    return dt

## OK KB - YOU NEED TO TEST THE AVERAGER WITH THE INPUT DATA

class DtypeTransposer:
    def __init__(self, cardidx, vis_source, values):
        # OK - now this is where it gets tricky and i wish I could refactor it properlty to get what I'm expecting
        self.nbeam = values.nbeams
        self.nrx = values.nrx
        self.dtype = get_transpose_dtype(values)
        self.mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(self.dtype)
        self.msgsize = self.dtype.itemsize
        self.nmsgs = 1
        log.info('Transposing %s with type=%s  size=%s NMSG=%s', self.dtype, self.mpi_dtype, self.msgsize, self.nmsgs)
        
        self.cardidx = cardidx
        self.vis_source = vis_source
        self.values = values
        
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        t_barrier = MPI.Wtime()
        comm.Barrier()
        t_start = MPI.Wtime()
        s_msg = [dtx,
                 (np2array(self.tx_counts),
                  np2array(self.tx_displacements)),
                 self.mpi_dtype]
            
        r_msg = [self.drx,
                 (np2array(self.rx_counts),
                  np2array(self.rx_displacements)),
                 self.mpi_dtype]
        comm.Alltoallv(s_msg, r_msg)
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
        self.dtype = get_transpose_dtype(info)
        self.mpi_dtype = MPI.BYTE
        values = info.values
                 
        #self.msgsize = self.dtype.itemsize
        if values.transpose_msg_bytes > 0:
            self.msgsize = min(values.transpose_msg_bytes, self.dtype.itemsize)
        else:
            self.msgsize = self.dtype.itemsize
                 
        self.displacement = self.dtype.itemsize
        self.nmsgs = (self.dtype.itemsize) // self.msgsize

        log.info('Transposing %s with type=%s  size=%s NMSG=%s', self.dtype, self.mpi_dtype, self.msgsize, self.nmsgs)
        
        self.values = info.values
        self.tx_counts = np.zeros(numprocs, np.int32)
        self.tx_displacements = np.zeros(numprocs, np.int32)
        self.rx_counts = np.zeros(numprocs, np.int32)
        self.rx_displacements = np.zeros(numprocs, np.int32)

    def all2all(self, dtx):
        nmsgs = self.nmsgs
        t_barrier = MPI.Wtime()
        log.debug('Rank %s waiting', comm.Get_rank())
        comm.Barrier()
        t_start = MPI.Wtime()
        t_wait = t_start - t_barrier
        log.debug('Rank %s finished barrier. Wait=%0.1f ms', comm.Get_rank(), t_wait*1000)

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
            comm.Alltoallv(s_msg, r_msg)
            
        t_end = MPI.Wtime()
        latency = (t_end - t_start)*1e3
        self.last_latency = latency
            
        #if rank == 0:
        #print(f'RANK0 Barrier wait: {(t_start - t_barrier)*1e3}ms Transpose latency = {latency}ms')

        #print(f'all2all COMPLETE {rank}/{numprocs} latency={latency} ms {t_start} {t_end}')

        return self.drx

class TransposeSender(ByteTransposer):
    def __init__(self, info):
        super().__init__(info)
        nbeam = self.nbeam
        self.tx_counts[:nbeam] = 1
        self.tx_displacements[:nbeam] = np.arange(nbeam, dtype=np.int32)
        self.drx = np.zeros((1), dtype=self.dtype) # Dummy for all toall make zero if possible

    def send(self, dtx):
        assert len(dtx) == self.nbeam
        assert dtx.dtype == self.dtype, f'Attempt to send invalid dtype. expected {self.dtype} but got {dtx.dtype}'
        return self.all2all(dtx)

class TransposeReceiver(ByteTransposer):
    def __init__(self, info):
        super().__init__(info)
        nbeam = self.nbeam
        nrx = self.nrx
        self.rx_counts[nbeam:] = 1 # receive same amount from every tx
        self.rx_displacements[nbeam:] = np.arange(nrx, dtype=np.int32)
        self.drx = np.zeros(nrx, dtype=self.dtype)
        self.dtx = np.zeros((1), dtype=self.dtype) # dummy buffer for sending TODO: make zero if possible

    def recv(self):
        return self.all2all(self.dtx)


def proc_rx(pipe_info): 
    '''
    Process 1 card per beam
    1 card = 6 FGPAs
    '''
    ccap = open_source(pipe_info)
    log.info('opened source')

    # tell all ranks about the headers
    all_hdrs = comm.allgather(ccap.fpga_headers)
    log.info('Got headers')
    
    info = MpiObsInfo(all_hdrs, pipe_info)
    log.info('got info')
    nbeam = pipe_info.nbeams
    cardidx = pipe_info.cardid
    values = pipe_info.values
    nrx = info.nrx
    nt = info.nt
    nant = info.nant
    nc = NCHAN*NFPGA
    npol_in = info.npol
    assert numprocs == nrx + nbeam

    dummy_packet = np.zeros((NCHAN*nbeam, ccap.merger.ntpkt_per_frame), dtype=ccap.merger.dtype)
    log.info('made packet %s this is a thing', dummy_packet.shape)
    rsout = os.path.join(pipe_info.values.outdir, f'rescale/b{ccap.block:02d}/c{ccap.card:02d}/') if pipe_info.values.cardcap_dir is not None else None

    averager = Averager(nbeam, nant, nc, nt, npol_in, values.vis_fscrunch, values.vis_tscrunch, REAL_DTYPE, CPLX_DTYPE, dummy_packet, values.flag_ants, rescale_output_path=rsout)
    log.info('made averager')
    
    transposer = TransposeSender(info)
    log.info('made transposer')

    # construct a typed list for numba - it's a bit of a pain but it needs to be done this way
    # Just need some types of the data so the list can be typed
    
    t_start = MPI.Wtime()

    log.info('Dummy packet shape %s dtype=%s. Averager output shape:%s dtype=%s', dummy_packet.shape, dummy_packet.dtype, averager.output.shape, averager.output.dtype)

    # Do dummy transpose to warm up and make connections
    transposer.send(averager.output)
    
    fid0 = ccap.start()

    # send fid0 to all beams
    fid0 = comm.bcast(fid0, root=pipe_info.rx_processor_rank0)

    pktiter = ccap.packet_iter()
    packets, fids = next(pktiter)

    averaged = averager.accumulate_packets(packets)
    averaged = averager.output
    info.fid0 = fid0
    
    start_fid = info.fid_of_block(1)
    log.info(f'rank={rank} fid0={fid0} {start_fid} {type(start_fid)}')
    best_avg_time = 1e6

    for ibuf, (packets, fids) in enumerate(pktiter):
        now = MPI.Wtime()
        read_time = now - t_start
        # so we have to put a dummy value in and add a separate flags array
        avg_start = MPI.Wtime()
        averaged = averager.accumulate_packets(packets)
        avg_end = MPI.Wtime()
        avg_time = avg_end - avg_start
        expected_fid = info.fid_of_block(ibuf) 

        best_avg_time = min(avg_time, best_avg_time)
        #print('RX times', read_time, avg_time, t_start, now, avg_start, avg_end)
        if avg_time*1e3 > 110:
            log.warning('Averaging time for cardidx=%s ibuf=%s was too long: %s ms best=%s', cardidx,ibuf, avg_time*1e3, best_avg_time*1e3)
            
        test_mode = False
        if test_mode:
            averaged['ics'].flat = cardidx #np.arange(averaged['ics'].size)
            averaged['cas'].flat = np.arange(averaged['cas'].size)
            averaged['vis'].flat = np.arange(averaged['vis'].size)

        transposer.send(averaged)
        if cardidx == 0:
            size_bytes = averaged.size * averaged.itemsize
            rate_gbps = size_bytes * 8/1e9/avg_time
            log.info('RANK0 transpose latency=%0.1fms accumulation time=%0.1fms last_nvalid=%d shape=%s dtype=%s, size=%s rate=%0.1fGbps',transposer.last_latency, avg_time*1e3, averager.last_nvalid, averaged.shape, averaged.dtype, size_bytes, rate_gbps)

        t_start = MPI.Wtime()
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
               'src_raj':pos.ra.deg,
               'src_dej':pos.dec.deg,
               'tstart':vis_source.tstart.utc.mjd,
               'tsamp':vis_source.inttime.to(u.second).value,
               'fch1':vis_source.fch1,
               'foff':vis_source.foff,
               'source_name':vis_source.target
        }

        self.fout = sigproc.SigprocFile(fname, 'wb', hdr)

    def write(self, beam_data):
        '''
        Writes data to the filterbank
        data has shape (nrx, nt,  nchan_per_rx) e.g. (36, 128, 24) 
        '''

        assert beam_data.dtype == np.float32, 'WE set nbits=32 for this filterbank, so if it isnt we need to do some work'
        dout = np.transpose(beam_data, [1, 0, 2])
#        if rank == 0:
            #print(dout.shape, dout.flatten().shape, dout.size, dout.mean(), dout.std())
        dout.tofile(self.fout.fin)

    def close(self):
        self.fout.fin.close()


class UvFitsFileSink:
    def __init__(self, obs_info):
        from craco.metadatafile import MetadataFile # Stupid seren doesn't have libopenblas on all nodes yet
        from craco.ccapfits2uvfits import get_antennas
        from craft.corruvfits import CorrUvFitsFile
        import scipy
        beamno = obs_info.pipe_info.beamid
        self.beamno = beamno
        self.obs_info = obs_info
        self.blockno = 0
        values = obs_info.values
        if values.fcm is None or values.metadata is None:
            log.info('Not writing UVFITS file as as FCM=%s or Metadata=%s not specified', values.fcm, values.metadata)
            self.uvout = None
            return
        
        fileout = os.path.join(values.outdir, f'b{beamno:02}.uvfits')
        self.fileout = fileout
        fcm = Parset.from_file(values.fcm)
        antennas = get_antennas(fcm)
        log.info('FCM %s contained %d antennas', values.fcm, len(antennas))
        info = obs_info
        fcent = info.fcent
        foff = info.foff * values.vis_fscrunch
        assert info.nchan % values.vis_fscrunch == 0, f'Fscrunch needs to divide nchan {info.nchan} {values.vis_fscrunch}'
        nchan = info.nchan // values.vis_fscrunch
        self.npol = 1 # card averager always sums polarisations
        npol = self.npol
        tstart = (info.tstart.value + info.inttime.to(u.day).value)
        self.total_nchan = nchan
        self.source_list = obs_info.sources().values()
        source_list = self.source_list
        log.info('UVFits sink opening file %s fcent=%s foff=%s nchan=%s npol=%s tstart=%s sources=%s nant=%d', fileout, fcent, foff, nchan, npol, tstart, source_list, len(antennas))

        self.uvout = CorrUvFitsFile(fileout,
                                    fcent,
                                    foff,
                                    nchan,
                                    npol,
                                    tstart,
                                    source_list,
                                    antennas)

        # create extra tables so we can fix it later on. if the file is not closed properly
        self.uvout.fq_table().writeto(fileout+".fq_table", overwrite=True)
        self.uvout.an_table(self.uvout.antennas).writeto(fileout+'.an_table', overwrite=True)
        self.uvout.su_table(self.uvout.sources).writeto(fileout+'.su_table', overwrite=True)
        self.uvout.hdr.totextfile(fileout+'.header', overwrite=True)

        with open(fileout+'.groupsize', 'w') as fout:
            fout.write(str(self.uvout.dtype.itemsize) + '\n')
            

    def write(self, vis_data):
        '''
        vis_data has len(nrx) and shape inner shape
        vishape = (nbl, vis_nc, vis_nt, 2) if np.int16 or
        or 
        vishape = (nbl, vis_nc, vis_nt) if np.complex64
        '''
        if self.uvout is None:
            return
        
        info = self.obs_info
        md = self.obs_info
        fid_start = info.fid_of_block(self.blockno)
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        assert nbl == info.nbl_flagged, f'Expected nbl={info.nbl_flagged} but got {nbl}'
        # TODO: Check timestamp convention for for FID and mjd.
        # I think this is right
        # fid_start goes up by NSAMP_PER_FRAME = 2048 every block
        # We'll calculate the same UVWs for everything in this block. A bit lazy but not rediculous
        fid_mid = fid_start + np.uint64(NSAMP_PER_FRAME//2)
        mjd = info.fid_to_mjd(fid_mid)
        sourceidx = md.source_index_at_time(mjd)
        uvw = md.uvw_at_time(mjd)
        antflags = md.antflags_at_time(mjd)
        dreshape = np.transpose(vis_data, (3,1,0,2)).reshape(vis_nt, nbl, self.total_nchan, self.npol) # should be [t, baseline, coarsechan*finechan]
        log.debug('Input data shape %s, output data shape %s', vis_data.shape, dreshape.shape)
        weights = np.ones((self.total_nchan, self.npol), dtype=np.float32)
        nant = info.nant
        inttime = info.inttime.to(u.second).value*info.vis_tscrunch
        assert NSAMP_PER_FRAME % vis_nt == 0
        samps_per_vis = np.uint64(NSAMP_PER_FRAME // vis_nt)

        # UV Fits files really like being in time order
        for itime in range(vis_nt):
            # FID is for the beginning of the block.
            # we might vis_nt = 2 and the FITS convention is to use the integraton midpoint
            fid_itime = fid_start + samps_per_vis // 2 + itime*samps_per_vis
            mjd = info.fid_to_mjd(fid_itime)
            log.debug('UVFITS block %s fid_start=%s fid_mid=%s info.nt=%s vis_nt=%s fid_itime=%s mjd=%s=%s inttime=%s', self.blockno, fid_start, fid_mid, info.nt, vis_nt, fid_itime, mjd, mjd.iso, inttime)
            for blinfo in info.baseline_iter():
                ia1 = blinfo.ia1
                ia2 = blinfo.ia2
                a1 = blinfo.a1
                a2 = blinfo.a2
                uvwdiff = uvw[ia1, :] - uvw[ia2, :]
                # TODO: channel-dependent weights - somehow
                dblk = dreshape[itime, blinfo.blidx, ...] # should be (nchan, npol)

                assert len(antflags) == info.nant_valid
                if antflags[ia1] or antflags[ia2]:
                    weights[:] = 0
                    dblk[:] = 0 # set output to zeros too, just so we can't cheat
                else:
                    weights[:] = 1
                    
                # fits convention has source index with starting value of 1
                fits_sourceidx = sourceidx + 1
                # put_data wants 0-based antenna numbers
                self.uvout.put_data(uvwdiff, mjd.value, a1-1, a2-1, inttime, dblk, weights, fits_sourceidx)
        
        self.uvout.fout.flush()
        log.debug(f'File size is {os.path.getsize(self.fileout)} blockno={self.blockno} ngroups={self.uvout.ngroups}')
        self.blockno += 1


    def close(self):
        print(f'Closing file {self.uvout}')
        if self.uvout is not None:
            self.uvout.close()

    def __del__(self):
        self.close()
        
def proc_beam(pipe_info):
    all_hdrs = comm.allgather([''])
    info = MpiObsInfo(all_hdrs, pipe_info)

    nbeam = pipe_info.nbeams
    values = pipe_info.values
    beamid = pipe_info.beamid
    nrx = info.nrx
    nt = info.nt
    nant = info.nant
    nc = NCHAN*NFPGA

    assert numprocs == nrx + nbeam, f'Invalid MPI setup numprocs={numprocs} nrx={nrx} nbeam={nbeam} expected {nrx + nbeam}'

    # OK - I need to gather all the headers from the data recivers
    # Beams don't kow headers, so we just send nothings
    if rank == 0:
        print(f'ObsInfo {info}')

    os.makedirs(values.outdir, exist_ok=True)

    transposer = TransposeReceiver(info)
    # Find first frame ID
    log.info('Recieving dummy transpose for warmup')
    transposer.recv()

    fid0 = 0
    fid0 = comm.bcast(fid0, root=pipe_info.rx_processor_rank0)
    info.fid0 = fid0
    
    cas_filterbank = FilterbankSink('cas',info)
    ics_filterbank = FilterbankSink('ics',info)
    vis_file = UvFitsFileSink(info)
    pipeline_sink = SearchPipelineSink(info)

    try:
        while True:
            beam_data = transposer.recv()
            cas_filterbank.write(beam_data['cas'])
            ics_filterbank.write(beam_data['ics'])
            vis_file.write(beam_data['vis'])
            pipeline_sink.write(beam_data['vis'])
    finally:
        print(f'Closing beam files for {beamid}')
        cas_filterbank.close()
        ics_filterbank.close()
        vis_file.close()
        pipeline_sink.close()


def parse_host_devices(hosts, devstr, devices):
    '''
    Get host devices from device string
    :hosts: list of host names
    :devstr: string like 'seren-01:1,seren-04:0,seren-5:0-1' of 'host:strrange' list of devices
    :devices: Tuple of devices that are normally allowed, e.g. (0,1)
    '''

    dead_cards = [dc.split(':') for dc in devstr.split(',')]
    host_cards = {}
    for h in hosts:
        my_devices = devices[:]
        my_bad_devices = [strrange(dc[1]) for dc in dead_cards if dc[0] == h]
        all_bad_devices = set()
        for bd in my_bad_devices:
            all_bad_devices.update(bd)


        host_cards[h] = tuple(sorted(set(devices) - all_bad_devices))
        
    return host_cards

def dump_rankfile(pipe_info, fpga_per_rx=3):
    from craco import mpiutil
    values = pipe_info.values
    fpga_per_rx = values.nfpga_per_rx
    hosts = pipe_info.hosts
    
    log.debug("Hosts %s", hosts)
    total_cards = len(values.block)*len(values.card)
    
    if values.max_ncards is not None:
        total_cards = min(total_cards, values.max_ncards)

    nrx = total_cards*len(values.fpga) // fpga_per_rx
    nbeams = values.nbeams
    nranks = nrx + nbeams
    ncards_per_host = (total_cards + len(hosts) - 1)//len(hosts) if values.ncards_per_host is None else values.ncards_per_host
        
    nrx_per_host = ncards_per_host
    nbeams_per_host = nbeams //len(hosts)
    log.info(f'Spreading {nranks} over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas and {nbeams} beams with {nbeams_per_host} per host')

    rank = 0
    host_search_beams = {}
    devices = (0,1)
    host_cards = parse_host_devices(hosts, values.dead_cards, devices)
    
    for beam in range(nbeams):
        hostidx = beam % len(hosts)
        assert hostidx < len(hosts), f'Invalid hostidx beam={beam} hostidx={hostidx} lenhosts={len(hosts)}'
        host = hosts[hostidx]
        slot = 0 # put on the U280 slot. If you put in slot1 it runs about 20% 
        core='1-2'
        devices = host_cards[host]
        this_host_search_beams = host_search_beams.get(host,[])
        host_search_beams[host] = this_host_search_beams
        if beam in values.search_beams:
            this_host_search_beams.append(beam)
            
        devid = None
        if beam in this_host_search_beams:
            devid = this_host_search_beams.index(beam)
            if devid not in devices:
                devid = None

        log.debug('beam %d devid=%s devices=%s host=%s this_host_search_beams=%s', beam, devid, devices, host, this_host_search_beams)
        rank_info = BeamRankInfo(beam, rank, host, slot, core, devid)
        pipe_info.beam_ranks.append(rank_info)
        rank += 1

    rxrank = 0
    cardno = 0
    for block in values.block:
        for card in values.card:
            cardno += 1
            if values.max_ncards is not None and cardno >= values.max_ncards + 1:
                break
            for fpga in values.fpga[::fpga_per_rx]:
                hostidx = rxrank // nrx_per_host
                hostrank = rxrank % nrx_per_host
                host = hosts[hostidx]
                slot = 1 # fixed because both cards are on NUMA=1
                # Put different FPGAs on differnt cores
                evenfpga = fpga % 2 == 0
                ncores = 10
                icore = (hostrank % 5)*2
                core = f'{icore}-{icore+3}' # let them be anywhere - need at least 3 cores / card
                core='0-9'
                slot = 1 # where the network cards are
                rank_info = ReceiverRankInfo(rxrank, rank, host, slot, core, block, card, fpga)
                pipe_info.receiver_ranks.append(rank_info)
                rank += 1
                rxrank += 1

        if values.dump_rankfile:
            with open(values.dump_rankfile, 'w') as fout:
                for rank_info in pipe_info.beam_ranks:
                    fout.write(rank_info.rank_file_str+'\n')
                for rank_info in pipe_info.receiver_ranks:
                    fout.write(rank_info.rank_file_str+'\n')

    
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from craco import search_pipeline
    
    cardcap_parser = cardcap.get_parser()
    pipeline_parser = search_pipeline.get_parser()
    parser = ArgumentParser(description='Run MPI pipeline', formatter_class=ArgumentDefaultsHelpFormatter, parents=[cardcap_parser, pipeline_parser], conflict_handler='resolve')
    
    parser.add_argument('--fcm', help='Path to FCM file for antenna positions')
    parser.add_argument('-m','--metadata', help='Path to schedblock metdata .json.gz file')
    parser.add_argument('--nfpga-per-rx', type=int, default=6, help='Number of FPGAS received by a single RX process')
    parser.add_argument('--vis-fscrunch', type=int, default=6, help='Amount to frequency average visibilities before transpose')
    parser.add_argument('--vis-tscrunch', type=int, default=1, help='Amount to time average visibilities before transpose')
    parser.add_argument('--ncards-per-host', type=int, default=None, help='Number of cards to process per host, helpful to match previous cardcap')
    parser.add_argument('--cardcap-dir', '-D', help='Local directory (per node?) to load cardcap files from, if relevant. If unspecified, just use files from the positional arguments')
    parser.add_argument('--transpose-msg-bytes', help='Size of the transpose block in bytes. If -1 do the whole block at once', type=int, default=-1)
    parser.add_argument('--search-beams', help='Beams to search', type=strrange, default=[])
    parser.add_argument('--dead-cards', help='List of dead cards to avoid. e.g.seren-01:1,seren-04:2', default='')
    parser.add_argument(dest='files', nargs='*')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    
    import platform

    class HostnameFilter(logging.Filter):
        hostname = platform.node()
        def filter(self, record):
            record.hostname = HostnameFilter.hostname
            record.rank = rank
            return True

    if values.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    FORMAT = '%(asctime)s [%(hostname)s:%(process)d] r%(rank)d %(module)s %(message)s'
    handler = logging.StreamHandler()
    handler.addFilter(HostnameFilter())
    handler.setFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

    try :
        pipe_info = MpiPipelineInfo(values)
        
        dump_rankfile(pipe_info) # always dump the rankfile - just need to do this to setup pipe_info
        if values.dump_rankfile:
            sys.exit(0)
            
        if pipe_info.is_beam_processor:
            proc_beam(pipe_info)
        else:
            proc_rx(pipe_info)

        log.info(f'Rank {rank}/{numprocs} complete. Waiting for everythign')
        #comm.Barrier()
    except:
        log.exception('Exception running pipeline')
        raise # raise it again so the interpreter dies and MPIRUN cleans it up

    
    if rank == 0:
        log.info('Pipeline complete')

#    raise StopIteration('Im raising a stop so it tears down the whole shebang. Perahps a bad idea. Maybe we should look at sending null form the receivers so the beams know theyre done and can shutdown clean')
    

if __name__ == '__main__':
    _main()
