#!/usr/bin/env python
"""
Runs MPI pipeline for CRACO.

Copyright (C) CSIRO 2022
"""
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
import numba
import glob
import craft.sigproc as sigproc
from craft.parset import Parset

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

# rank ordering
# [ beam0, beam1, ... beamN-1 | rx0, rx1, ... rxM-1]

REAL_DTYPE = np.float32 # Transpose/averaging type for real types
CPLX_DTYPE = np.complex64 # Tranaspose/averagign dtype for complex types  can also be a real type, like np.int16 and somethign willl add an extra dimension

class MpiPipelineInfo:
    def __init__(self, values):

        if values.max_ncards <= 0:
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
        bproc = self.world_rank < self.nbeams
        return bproc

    @property
    def beamid(self):
        assert self.is_beam_processor, 'Requested beam ID of non beam processor'
        return self.world_rank

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


    @property
    def nbeams(self):
        return self.pipe_info.nbeams

    @property
    def nt(self):
        return self.pipe_info.nt

    @property
    def nrx(self):
        return self.pipe_info.nrx
    
    @property
    def target(self):
        return self.main_merger.target

    def fid_to_mjd(self, fid):
        return self.main_merger.fid_to_mjd(fid)

    @property
    def vis_fscrunch(self):
        return self.values.vis_fscrunch

    @property
    def vis_tscrunch(self):
        return self.values.vis_tscrunch
        
    @property
    def fid0(self):
        return self.__fid0

    @fid0.setter
    def fid0(self, fid):
        self.__fid0 = np.uint64(fid)

    @property
    def tstart(self):
        '''
        Returns astropy time given fid0
        We assume we start on the frame after fid0
        '''
        assert self.__fid0 is not None, 'First frame ID must be set before we can calculate tstart'
        fid_first = self.fid_of_block(0)
        return self.main_merger.ccap[0].time_of_frame_id(fid_first)

    def fid_of_block(self, iblk):
        return self.fid0 + np.uint64(iblk + 1)*np.uint64(NSAMP_PER_FRAME)

    @property
    def nchan(self):
        return self.main_merger.nchan

    @property
    def npol(self):
        return self.main_merger.npol

    @property
    def fch1(self):
        return self.main_merger.fch1

    @property
    def fcent(self):
        return self.main_merger.fcent

    @property
    def foff(self):
        return self.main_merger.foff

    def skycoord(self, beam):
        '''
        TODO: Fill this in
        '''
        return SkyCoord('15h30m00s','-30d00m00s')

    @property
    def inttime(self):
        return self.main_merger.inttime

    @property
    def nt(self):
        '''
        Return number of samples per frame
        '''
        return self.main_merger.nt_per_frame

    @property
    def nant(self):
        return self.main_merger.nant

    def __str__(self):
        m = self.main_merger
        s = f'fch1={m.fch1} foff={m.foff} nchan={m.nchan} nant={m.nant} inttime={m.inttime}'
        return s


def get_transpose_dtype(values):
    nbeam = values.nbeams
    nt = values.nt
    nant = values.nant
    nc = NCHAN*NFPGA
    vis_fscrunch = values.vis_fscrunch
    vis_tscrunch = values.vis_tscrunch
    npol = values.npol
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
            
        if rank == 0:
            print(f'RANK0 Barrier wait: {(t_start - t_barrier)*1e3}ms Transpose latency = {latency}ms')

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
        #comm.Barrier()
        t_start = MPI.Wtime()
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
            
        if rank == 0:
            print(f'RANK0 Barrier wait: {(t_start - t_barrier)*1e3}ms Transpose latency = {latency}ms')

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

    # tell all ranks about the headers
    all_hdrs = comm.allgather(ccap.fpga_headers)
    
    info = MpiObsInfo(all_hdrs, pipe_info)
    nbeam = pipe_info.nbeams
    cardidx = pipe_info.cardid
    values = pipe_info.values
    nrx = info.nrx
    nt = info.nt
    nant = info.nant
    nc = NCHAN*NFPGA
    npol = 1 if values.pol_sum else 2
    assert numprocs == nrx + nbeam

    dummy_packet = np.zeros((ccap.merger.npackets_per_frame), dtype=ccap.merger.dtype)
    rsout = os.path.join(pipe_info.values.outdir, f'rescale/b{ccap.block:02d}/c{ccap.card:02d}/')
    averager = Averager(nbeam, nant, nc, nt, npol, values.vis_fscrunch, values.vis_tscrunch, REAL_DTYPE, CPLX_DTYPE, dummy_packet, values.exclude_ants, rescale_output_path=rsout)
    transposer = TransposeSender(info)

    # construct a typed list for numba - it's a bit of a pain but it needs to be done this way
    # Just need some types of the data so the list can be typed
    
    t_start = MPI.Wtime()

    log.debug('Dummy packet shape %s dtype=%s. Averager output shape:%s dtype=%s', dummy_packet.shape, dummy_packet.dtype, averager.output.shape, averager.output.dtype)

    # get initial data to setup scaling and find starting FID
    packets, fids = next(ccap.packet_iter())
    averaged = averager.accumulate_packets(packets)
    averaged = averager.output
    maxfid = max([0 if fid is None else fid for fid in fids])
    all_maxfid = comm.allreduce(maxfid, MPI.MAX)
    info.fid0 = all_maxfid
    start_fid = info.fid_of_block(0)
    log.info(f'rank={rank} maxfid={maxfid} allmaxfid={all_maxfid} myfid={maxfid} {type(maxfid)} {type(all_maxfid)} {start_fid} {type(start_fid)}')


    for ibuf, (packets, fids) in enumerate(ccap.packet_iter(start_fid)):
        now = MPI.Wtime()
        read_time = now - t_start
        # so we have to put a dummy value in and add a separate flags array
        avg_start = MPI.Wtime()
        averaged = averager.accumulate_packets(packets)
        averaged = averager.output
        avg_end = MPI.Wtime()
        avg_time = avg_end - avg_start
        expected_fid = info.fid_of_block(ibuf)
        for ifpga, (pkt, fid) in enumerate(zip(packets, fids)):
            if pkt is not None:
                diff = int(expected_fid) - int(fid)
                assert diff==0, f'Invalid fid. Expected {expected_fid} but got {fid} diff={diff} for rank={rank} ibuf={ibuf} card={cardidx} ifpga={ifpga} start_fid={start_fid} all_maxfid={all_maxfid} maxfid={maxfid}'
            
        #print('RX times', read_time, avg_time, t_start, now, avg_start, avg_end)
        if avg_time*1e3 > 50:
            log.warning('Averaging time for %s was too long: %s ms', cardidx, avg_time*1e3)
            
        test_mode = False
        if test_mode:
            averaged['ics'].flat = cardidx #np.arange(averaged['ics'].size)
            averaged['cas'].flat = np.arange(averaged['cas'].size)
            averaged['vis'].flat = np.arange(averaged['vis'].size)

        transposer.send(averaged)
        t_start = MPI.Wtime()
        if ibuf == values.num_msgs -1:
            raise ValueError('Stopped')


class FilterbankSink:
    def __init__(self, prefix, beamid, vis_source, values):
        fname = os.path.join(values.outdir, f'{prefix}_b{beamid:02d}.fil')
        pos = vis_source.skycoord(beamid)
        hdr = {'nbits':32,
               'nchans':vis_source.nchan,
               'nifs':vis_source.npol,
               'src_raj':pos.ra.deg,
               'src_dej':pos.dec.deg,
               'tstart':vis_source.tstart.utc.mjd,
               'tsamp':vis_source.inttime,
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
    def __init__(self, beamno, obs_info):
        from craco.metadatafile import MetadataFile # Stupid seren doesn't have libopenblas on all nodes yet
        from craco.ccapfits2uvfits import get_antennas
        from craft.corruvfits import CorrUvFitsFile
        import scipy
        self.c = scipy.constants.c

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
        tstart = info.tstart.value + info.inttime/3600./24.
        self.total_nchan = nchan
        self.md = MetadataFile(values.metadata)
        self.source_list = self.md.sources(self.beamno).values()
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

        print(self.uvout.dtype, self.uvout.dtype.itemsize)
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
        md = self.md
        fid_start = info.fid_of_block(self.blockno)
        nrx, nbl, vis_nc, vis_nt = vis_data.shape[:4]
        # TODO: Check timestamp convention for for FID and mjd.
        # I think this is right
        fid_mid = fid_start + info.nt // 2
        print('*'*10, fid_start, fid_mid, info)
        mjd = info.fid_to_mjd(fid_mid)
        sourceidx = md.source_index_at_time(mjd.value)
        sourcename = md.source_name_at_time(mjd.value)
        uvw = md.uvw_at_time(mjd.value)[:, self.beamno, :] /self.c # UVW in seconds
        antflags = md.flags_at_time(mjd.value)
        print(f'Input data shape {vis_data.shape}')
        dreshape = np.transpose(vis_data, (3,1,0,2)).reshape(vis_nt, nbl, self.total_nchan, self.npol) # should be [t, baseline, coarsechan*finechan]
        log.debug('Input data shape %s, output data shape %s', vis_data.shape, dreshape.shape)
        weights = np.ones((self.total_nchan, self.npol), dtype=np.float32)
        nant = info.nant
        inttime = info.inttime


        # UV Fits files really like being in time order
        for itime in range(vis_nt):
            blidx = 0
            mjd = info.fid_to_mjd(fid_start + itime)
            for ia1 in range(nant):
                for ia2 in range(ia1, nant):
                    uvwdiff = uvw[ia1, :] - uvw[ia2, :]
                    # TODO: channel-dependent weights - somehow
                    dblk = dreshape[itime, blidx, ...] # should be (nchan, npol)
                    
                    if antflags[ia1] or antflags[ia2]:
                        weights[:] = 0
                        dblk[:] = 0 # set output to zeros too, just so we can't cheat
                    else:
                        weights[:] = 1
                        
                    self.uvout.put_data(uvwdiff, mjd.value, ia1, ia2, inttime, dblk, weights, sourceidx)
                    blidx += 1
        
        self.uvout.fout.flush()
        print(f'File size is {os.path.getsize(self.fileout)} blockno={self.blockno} ngroups={self.uvout.ngroups}')
        self.blockno += 1


    def close(self):
        print(f'Closing file {self.uvout}')
        if self.uvout is not None:
            self.uvout.close()
        
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
    npol = 1 if values.pol_sum else 2

    assert numprocs == nrx + nbeam, f'Invalid MPI setup numprocs={numprocs} nrx={nrx} nbeam={nbeam} expected {nrx + nbeam}'

    # OK - I need to gather all the headers from the data recivers
    # Beams don't kow headers, so we just send nothings
    if rank == 0:
        print(f'ObsInfo {info}')
        
    transposer = TransposeReceiver(info)
    os.makedirs(values.outdir, exist_ok=True)

    # Find first frame ID
    firstfid = comm.allreduce(0, op=MPI.MAX)
    info.fid0 = firstfid
    
    cas_filterbank = FilterbankSink('cas', beamid, info, values)
    ics_filterbank = FilterbankSink('ics', beamid, info, values)
    vis_file = UvFitsFileSink(beamid, info)

    try:
        while True:
            beam_data = transposer.recv()
            cas_filterbank.write(beam_data['cas'])
            ics_filterbank.write(beam_data['ics'])
            vis_file.write(beam_data['vis'])
    finally:
        print(f'Closing beam files for {beamid}')
        cas_filterbank.close()
        ics_filterbank.close()
        vis_file.close()


def dump_rankfile(values, fpga_per_rx=3):
    from craco import mpiutil
    hosts = mpiutil.parse_hostfile(values.hostfile)
    log.debug("Hosts %s", hosts)
    total_cards = len(values.block)*len(values.card)
    
    if values.max_ncards != 0:
        total_cards = min(total_cards, values.max_ncards)

    nrx = total_cards*len(values.fpga) // fpga_per_rx
    nbeams = values.nbeams
    nranks = nrx + nbeams
    ncards_per_host = (total_cards + len(hosts) - 1)//len(hosts) if values.ncards_per_host is None else values.ncards_per_host
        
    nrx_per_host = ncards_per_host
    nbeams_per_host = (nbeams + len(hosts) - 1)//len(hosts)
    log.info(f'Spreading {nranks} over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas and {nbeams} beams with {nbeams_per_host} per host')

    rank = 0
    with open(values.dump_rankfile, 'w') as fout:
        # add all the beam processes
        for beam in range(nbeams):
            hostidx = rank // nbeams_per_host
            hostrank = rank % nbeams_per_host
            host = hosts[hostidx]
            slot = 0 # put on the U280 slot. If you put in slot1 it runs about 20% 
            core='5-6'
            s = f'rank {rank}={host} slot={slot}:{core} # Beam {beam}\n'
            fout.write(s)
            rank += 1

        rxrank = 0
        cardno = 0
        for block in values.block:
            for card in values.card:
                cardno += 1
                if values.max_ncards != 0 and cardno >= values.max_ncards + 1:
                    break
                for fpga in values.fpga[::fpga_per_rx]:
                    hostidx = rxrank // nrx_per_host
                    hostrank = rxrank % nrx_per_host
                    host = hosts[hostidx]
                    slot = 1 # fixed because both cards are on NUMA=1
                    # Put different FPGAs on differnt cores
                    evenfpga = fpga % 2 == 0
                    core = rxrank % 10
                    slot = 1
                    s = f'rank {rank}={host} slot={slot}:{core} # Block {block} card {card} fpga {fpga}\n'
                    fout.write(s)
                    rank += 1
                    rxrank += 1
    
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from craft.cmdline import strrange
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    cardcap.add_arguments(parser)
    parser.add_argument('--fcm', help='Path to FCM file for antenna positions')
    parser.add_argument('-m','--metadata', help='Path to schedblock metdata .json.gz file')
    parser.add_argument('--nfpga-per-rx', type=int, default=6, help='Number of FPGAS received by a single RX process')
    parser.add_argument('--vis-fscrunch', type=int, default=6, help='Amount to frequency average visibilities before transpose')
    parser.add_argument('--vis-tscrunch', type=int, default=1, help='Amount to time average visibilities before transpose')
    parser.add_argument('--ncards-per-host', type=int, default=None, help='Number of cards to process per host, helpful to match previous cardcap')
    parser.add_argument('--cardcap-dir', '-D', help='Local directory (per node?) to load cardcap files from, if relevant. If unspecified, just use files from the positional arguments')
    parser.add_argument('--outdir', '-O', help='Directory to write outputs to', default='.')
    parser.add_argument('--transpose-msg-bytes', help='Size of the transpose block in bytes. If -1 do the whole block at once', type=int, default=-1)
    parser.add_argument('--exclude-ants', help='Antenna numbers to exclude e.g. 1,2,3,35-36', type=strrange, default='')
    parser.add_argument(dest='files', nargs='*')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    pipe_info = MpiPipelineInfo(values)

    if values.dump_rankfile:
        dump_rankfile(pipe_info.values, values.nfpga_per_rx)
        sys.exit(0)

    if pipe_info.is_beam_processor:
        proc_beam(pipe_info)
    else:
        proc_rx(pipe_info)

    comm.Barrier()
    print(f'Rank {rank}/{numprocs} complete')

                  
    

if __name__ == '__main__':
    _main()
