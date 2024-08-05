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
from craco import mpiutil
from collections import namedtuple, OrderedDict
from craft.cmdline import strrange
from mpi4py import MPI
from craco.mpi_tracefile import MpiTracefile


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"


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

class MpiAppInfo:
    '''
    Assume we have 5 types of process:
    RX - receives data from cards -  per card
    BEAMTRAN - receives beam transpose. 1 per beam.
    BEAMPROC - processes beam data. 1 per beam
    PLANNER - creates plan and sends async to beamproc - 1 per beam
    BEAM_CAND - recieves candidates from BEAMPROC - 1 per beam
    CAND_MGR - consolidate candidates from BEAM_CAND  - 1 per application

    A " beam chain" is BEAMROC->PLANNER->BEAM_CAND

    This class creates a few useful communicators:

    world_comm = world communicator
    app_comm = communicator for all ranks in the app we're in
    rx_beam_comm = communicator for RX and BEAMPROC ranks. Ranks that aren't that have None
    beam_chain_comm = communicator for BEAMPROC, PLANNER and BEAM_CAND ranks, split by beamid. rank[0] is the
    BEAMPROC, ranks[1] is the PLANNER and rank[2] is the BEAM_CAND
    cand_comm = communicator for BEAM_CAND and BEAM_MGR ranks. Rank[0] is the BEAM_MGR, the remainder are the 
    beam processor (rank=beamid+1)
    '''

    RX_APPNUM = 0
    BEAMTRAN_APPNUM = 1
    BEAMPROC_APPNUM = 2
    PLANNER_APPNUM = 3
    CAND_MGR_APPNUM = 4 # want manager before beam cand processor so it ends up as rank 0 in the cand_comm
    BEAM_CAND_APPNUM = 5

    # inside a beam chain
    BEAMTRAN_RANK = 0
    BEAMPROC_RANK = 1
    PLANNER_RANK = 2
    CANDPROC_RANK = 3


 
    def __init__(self, pipe_info, proctype):
        self.pipe_info = pipe_info
        self.world = MPI.COMM_WORLD
        comm = self.world
        self.world_rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.world_comm = comm

        appnum = self.app_num

        # Communicator for my app
        self.app_comm = comm.Split(appnum, self.world_rank)
        self.app_rank = self.app_comm.Get_rank()
        self.app_size = self.app_comm.Get_size()

        rxbeam_colour = 0 if self.is_rx_processor or self.is_beam_transposer else -1

        # communicator for the data to from RX to beam
        log.info('Splitting rxbeam colour=%d rank=%s', rxbeam_colour, self.world_rank)
        rx_beam_comm = comm.Split(rxbeam_colour, self.world_rank)
        self.rx_beam_comm = rx_beam_comm if rxbeam_colour >= 0 else None

        # Beam chain communicator - [0]=rx, [1] = planner, [2] = cand_processor
        # chain is coloured by beamid
        beam_chain_colour = self.beamid if self.is_in_beam_chain else -1
        beam_chain_comm = comm.Split(beam_chain_colour, self.world_rank)
        self.beam_chain_comm = beam_chain_comm if beam_chain_colour >= 0 else None
        
        if self.is_in_beam_chain:
            bcrank = self.beam_chain_comm.Get_rank()
            if self.is_beam_transposer:
                assert bcrank == MpiAppInfo.BEAMTRAN_RANK
            elif self.is_beam_processor:
                assert bcrank == MpiAppInfo.BEAMPROC_RANK
            elif self.is_planner_processor:
                assert bcrank == MpiAppInfo.PLANNER_RANK
            elif self.is_cand_processor:
                assert bcrank == MpiAppInfo.CANDPROC_RANK

        # candidate communicator.
        # rank=[0] is manager,
        # ranks[1..nbeam] = beam cand processors
        is_cand = self.is_cand_processor or self.is_cand_manager
        cand_color = 0 if is_cand else -1
        cand_comm = comm.Split(cand_color, self.world_rank)
        self.cand_comm = cand_comm if cand_color >= 0 else None

        log.debug('MPIAppInfo: type=%s, World=%d/%d appnum=%d rank=%d/%d. Colours: Rxbeam %d beam_chain: %d cand: %d',
                  proctype, self.world_rank, self.world_size, self.app_num, self.app_rank, self.app_size,
                  rxbeam_colour, beam_chain_colour, cand_color)    
 
        if self.is_cand_manager:
            assert self.app_size == 1, f'Cand manager should have 1 app. Has {self.app_size}'
            cand_rank = self.cand_comm.Get_rank()
            assert cand_rank == 0, f'Cand manager should have 0 rank. Was {cand_rank}'       
        
    @property
    def app_num(self):
        '''
        So if you're wodnering why all this stupid Appnum stuff is flying around, it's because there's meant to be a useful MPIRUN
        features where you can specifiy different apps serparate by a :. But it doesn't work because if you do -map-by ppr:X:socket
        then it applies the same to all the apps - which means you can't overload more cards than beams, for example.
        Maybe I need to submit a ticket to OpenMPI.
        In the meantime I've refactored it to work with a rankfile again - all the apps live in the same rankfile
        and you work out which appnum you are based on your world rank and other stuff.
        '''
        #appnum = MPI.COMM_WORLD.Get_attr(MPI.APPNUM)  # we no longer use apps
        rankinfo = self.rank_info
        return rankinfo.APP_ID
    
    @property
    def rank_info(self):
        return self.pipe_info.all_ranks[self.world_rank]
        
    @property
    def is_rx_processor(self):
        return self.app_num == MpiAppInfo.RX_APPNUM
    
    @property
    def is_beam_transposer(self):
        return self.app_num == MpiAppInfo.BEAMTRAN_APPNUM
    
    @property
    def is_beam_processor(self):
        return self.app_num == MpiAppInfo.BEAMPROC_APPNUM

    @property
    def is_planner_processor(self):
        return self.app_num == MpiAppInfo.PLANNER_APPNUM
    
    @property
    def is_cand_processor(self):
        return self.app_num == MpiAppInfo.BEAM_CAND_APPNUM
    
    @property
    def is_cand_manager(self):
        return self.app_num == MpiAppInfo.CAND_MGR_APPNUM

    @property
    def is_in_beam_chain(self):
        return self.is_beam_transposer or self.is_beam_processor or self.is_planner_processor or self.is_cand_processor
    
    @property
    def proc_name(self):
        '''
        Returns the RankInfo class name without the 'Rankinfo' part
        '''
        info = self.rank_info
        n = info.__class__.__name__.replace('RankInfo','')
        return n
    
    @property
    def proc_labels(self):
        '''
        Returns some kindof labelling of this process as a space-delimited string
        '''
        return self.rank_info.labels

    @property
    def beamid(self):
        '''
        Returns BEAMID in range 0-35 for beamid. 
        '''
        assert self.is_in_beam_chain, 'Only beam chain processors have beamids'
        b = self.app_rank
        return b

    @property
    def cardid(self):
        assert self.is_rx_processor, 'Must be a card processor to have a cardid'
        c = self.app_rank
        return c


class ReceiverRankInfo(namedtuple('ReceiverRankInfo', ['rxid','rank','host','slot','core','block','card','fpga'])):
    APP_ID = MpiAppInfo.RX_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Block {self.block} card {self.card} fpga {self.fpga}'
        return s
    
    def __str__(self):
        return f"Rx {self.block}/{self.card}/{self.fpga}"

    @property
    def labels(self):
        return f'block{self.block} card{self.card} fpga{self.fpga} {self.host}'
    
class BeamTranRankInfo(namedtuple('BeamTranRankInfo', ['beamid','rank','host','slot','core'])):
    APP_ID = MpiAppInfo.BEAMTRAN_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Beam {self.beamid} transpose receiver '
        return s
    
    def __str__(self):
        return f'BeamTran {self.beamid}'
    
    @property
    def labels(self):
        return f'beam{self.beamid} {self.host}'
    
class BeamProcRankInfo(namedtuple('BeamProcRankInfo', ['beamid','rank','host','slot','core','xrt_device_id'])):
    APP_ID = MpiAppInfo.BEAMPROC_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Beam {self.beamid} processor xrtdevid={self.xrt_device_id}'
        return s
    
    @property
    def labels(self):
        return f'beam{self.beamid} {self.host}'

class PlannerRankInfo(namedtuple('PlannerRankInfo', ['beamid','rank','host','slot','core'])):
    APP_ID = MpiAppInfo.PLANNER_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Beam {self.beamid} Planner'
        return s
    
    @property
    def labels(self):
        return f'beam{self.beamid} {self.host}'

class BeamCandRankInfo(namedtuple('BeamCandRankInfo', ['beamid','rank','host','slot','core'])):
    APP_ID = MpiAppInfo.BEAM_CAND_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Beam {self.beamid} Cand processor'
        return s
    
    @property
    def labels(self):
        return f'beam{self.beamid} {self.host}'


class CandMgrRankInfo(namedtuple('CandMgrRankInfo', ['rank','host','slot','core'])):
    APP_ID = MpiAppInfo.CAND_MGR_APPNUM
    @property
    def rank_file_str(self):
        s = f'rank {self.rank}={self.host} slot={self.slot}:{self.core} # Cand manager'
        return s
    
    @property
    def labels(self):
        return f'{self.host}'
    

def populate_ranks(pipe_info, total_cards):
    from craco import mpiutil
    values = pipe_info.values
    fpga_per_rx = values.nfpga_per_rx
    hosts = pipe_info.hosts
    
    log.debug("Hosts %s", hosts)
    nbeams = values.nbeams
    nranks = total_cards + nbeams*3 + 1
    ncards_per_host = (total_cards + len(hosts) - 1)//len(hosts) if values.ncards_per_host is None else values.ncards_per_host
        
    nrx_per_host = ncards_per_host
    nbeams_per_host = nbeams //len(hosts)
    log.info(f'Spreading {nranks} over {len(hosts)} hosts {len(values.block)} blocks * {len(values.card)} * {len(values.fpga)} fpgas and {nbeams} beams with {nbeams_per_host} per host')

    rank = 0
    rxrank = 0
    cardno = 0
    ncores_per_socket = 16
    nslots_per_host = 2

    for block in values.block:
        for card in values.card:
            cardno += 1
            if cardno >=total_cards + 1:
                break
            for fpga in values.fpga[::fpga_per_rx]:
                hostidx = rxrank // nrx_per_host
                hostrank = rxrank % nrx_per_host                
                host = hosts[hostidx]
                ncores_per_proc = 1
                slot = (hostrank // nslots_per_host) % nslots_per_host
                icore = hostrank % nslots_per_host
                #icore = (slotrank*ncores_per_proc) % ncores_per_socket 
                #core='0-9'
                core = f'{icore}-{icore+ncores_per_proc-1}'
                rank_info = ReceiverRankInfo(rxrank, rank, host, slot, core, block, card, fpga)
                pipe_info.add_rank(rank_info)
                rank += 1
                rxrank += 1

    host_search_beams = {}
    devices = (0,1)
    host_cards = parse_host_devices(hosts, values.dead_cards, devices)

    pipe_info.add_rank(CandMgrRankInfo(rank, 'skadi-00', slot, core))
    rank += 1
    
    for beam in range(nbeams):
        hostidx = beam % len(hosts)
        assert hostidx < len(hosts), f'Invalid hostidx beam={beam} hostidx={hostidx} lenhosts={len(hosts)}'
        host = hosts[hostidx]
        nslots = 2
        slot = beam % nslots  # put on the U280 slot. If you put in slot1 it runs about 20% 
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
        pipe_info.add_rank(BeamTranRankInfo(beam, rank, host, slot, core='13'))
        rank += 1
        pipe_info.add_rank(BeamProcRankInfo(beam, rank, host, slot, core='9-12', xrt_device_id=devid)) # extra cores for prepare step multi-threading
        rank += 1
        pipe_info.add_rank(PlannerRankInfo(beam, rank, host, slot, core='14'))
        rank += 1
        pipe_info.add_rank(BeamCandRankInfo(beam, rank, host, slot, core='15'))
        rank += 1

    if values.dump_rankfile:
        with open('mpipipeline.rank', 'w') as fout:
            for rank_info in pipe_info.all_ranks.values():
                fout.write(rank_info.rank_file_str + '\n')
        with open('beam.rank', 'w') as fout:
            for rank_info in pipe_info.beam_ranks:
                fout.write(rank_info.rank_file_str+'\n')
        with open('rx.rank', 'w') as fout:
            for rank_info in pipe_info.receiver_ranks:
                fout.write(rank_info.rank_file_str+'\n')

                        
class MpiPipelineInfo:
    def __init__(self, values):
        '''
        TODO - integrate this in with a nice way of doing dump hostfile
        '''
        self.values = values
        self.hosts = mpiutil.parse_hostfile(values.hostfile)
        #self.beam_ranks = []
        #self.receiver_ranks = []
        #self.planner_ranks = []
        #self.beam_cand_ranks = []
        #self.beam_mgr_ranks =[]
        self.all_ranks = OrderedDict()
        
        ncards = len(values.block)*len(values.card)
        
        if values.max_ncards is not None:
            ncards = min(ncards, values.max_ncards)

        nrx = ncards*len(values.fpga)//values.nfpga_per_rx
        self.nrx = nrx
        self.ncards = ncards

        assert ncards == nrx, 'Im not sure we support receiving a weird number of FPGAs per process anymore. We might Im just not sure.'

        # yuck. This is just yuk.
        if values.beam is None:
            values.nbeams = 36
        else:
            assert 0<= values.beam < 36, f'Invalid beam {values.beam}'
            values.nbeams = 1
        
        self.nbeams = values.nbeams
        self.world_comm = MPI.COMM_WORLD
        world = self.world_comm
        self.world_rank = world.Get_rank()
        populate_ranks(self, ncards)      

        self.mpi_app = MpiAppInfo(self, values.proc_type)
        MpiTracefile.instance().add_metadata(process_name=self.mpi_app.proc_name,
                                                       process_labels=self.mpi_app.proc_labels,
                                                       process_sort_index=self.world_rank)


    def add_rank(self, rankinfo):
        self.all_ranks[rankinfo.rank] = rankinfo

    def get_ranks_for_app(self, appid):
        return list(filter(lambda r:r.APP_ID == appid, self.all_ranks.values()))

    @property
    def beam_ranks(self):
        return self.get_ranks_for_app(MpiAppInfo.BEAMPROC_APPNUM)
    
    @property
    def receiver_ranks(self):
        return self.get_ranks_for_app(MpiAppInfo.RX_APPNUM)

    @property
    def rx_comm(self):
        assert self.mpi_app.is_rx_processor
        return self.mpi_app.app_comm
    
    @property
    def cardid(self):
        return self.mpi_app.cardid
    
    @property
    def beamid(self):
        return self.mpi_app.beamid


    @property
    def is_beam_processor(self):
        ''' 
        First nbeams ranks are beam processors
        Send nrx ranks are RX processor
        '''
        return self.mpi_app.is_beam_processor

    @property
    def rx_processor_rank0(self):
        raise NotImplemented()
        #return self.nbeams # Rank0 for rx processor is

    @property
    def beamid(self):
        return self.mpi_app.beamid

    def beam_rank_info(self, beamid):
        '''
        Returns the rank info for the given beam
        '''
        
        rank_info = next(filter(lambda info: info.beamid == beamid, self.beam_ranks))
        return rank_info

    @property
    def is_rx_processor(self):
        return self.mpi_app.is_rx_processor

    @property
    def cardid(self):
        return self.mpi_app.cardid


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
