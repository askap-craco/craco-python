#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
from mpi4py import MPI

import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.candidate_writer import CandidateWriter
from craco.mpi_tracefile import MpiTracefile
from craco.tracing import tracing
import mpi4py.util.dtlib
from craco.mpiutil import np2array
from astropy.time import Time
from astropy import units as u

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

MAX_NCAND_OUT = 8

class MpiCandidateBuffer:
    def __init__(self, max_ncand=8192):        
        self.cands = np.zeros(max_ncand, dtype=CandidateWriter.out_dtype)
        self.mpi_dtype = mpi4py.util.dtlib.from_numpy_dtype(CandidateWriter.out_dtype)
        self.mpi_msg = [self.cands, max_ncand, self.mpi_dtype]
        self.trace_file = MpiTracefile.instance()

    def gather(self):
        self.comm.Gatherv(self.s_msg, self.r_msg)
        valid = self.cands['snr'] > 0
        outcands = self.cands[valid]
        self.trace_cands(outcands)
        return outcands
    
    def send(self, ncand):
        assert 0<= ncand <= len(self.cands)
        mpi_msg = [self.cands, ncand, self.mpi_dtype]
        self.comm.Send(mpi_msg, dest=self.destrank)
        self.trace_cands(self.cands[:ncand])

    def recv(self):
        status = MPI.Status()
        self.comm.Recv(self.mpi_msg, source=self.rxrank, status=status)
        ncand = status.Get_count(self.mpi_dtype)
        candout = self.cands[:ncand]
        self.trace_cands(candout)
        return candout
    
    def trace_cands(self, cands):
        now = Time.now()
        diffs = now - Time(cands['mjd'], format='mjd', scale='utc')        
        latency = 0 if len(cands) == 0 else diffs.to(u.millisecond).max().value
        maxsnr = 0 if len(cands) == 0 else cands['snr'].max()
        log.debug('Tracing cands %d %f %f', len(cands), maxsnr, latency)
        self.trace_file += tracing.CounterEvent('Candidates', args={'ncands':len(cands)},ts=None)
        self.trace_file += tracing.CounterEvent('SNR', args={'maxsnr':maxsnr},ts=None)
        self.trace_file += tracing.CounterEvent('Latency',args={'latency':latency},ts=None)
        
    
    @staticmethod
    def for_tx(comm, destrank):
        cbuf = MpiCandidateBuffer()
        cbuf.comm = comm
        cbuf.destrank = destrank
        return cbuf


    @staticmethod
    def for_rx(comm, rxrank):
        cbuf = MpiCandidateBuffer()
        cbuf.comm = comm
        cbuf.rxrank = rxrank
        return cbuf

    

    @staticmethod
    def for_beam_manager(comm):
        '''
        Creates candidate buffer for a beam manager
        
        Recives MAX_NCAND_OUT candiites for nbeams beams
        saves the appropriate messages in MPI format
        so that 'gather' can be called
        '''
        nranks = comm.Get_size()
        nbeams = nranks - 1 # 1 rank is the manager - me?
        
        ncand = nbeams*MAX_NCAND_OUT
        cands = MpiCandidateBuffer(ncand)
        
        tx_cands = 0

        s_msg = [cands.cands,
                tx_cands,     
                cands.mpi_dtype]
        
        rx_counts = np.ones(nranks, dtype=np.int32)*MAX_NCAND_OUT
        rx_counts[0] = 0
        rx_displacements = np.zeros(nranks, dtype=np.int32)
        rx_displacements[1:] = np.arange(nranks-1)*MAX_NCAND_OUT
        r_msg = [cands.cands,
                np2array(rx_counts),
                np2array(rx_displacements),
                cands.mpi_dtype]
        
        cands.comm = comm
        cands.r_msg = r_msg
        cands.s_msg = s_msg
        return cands

    @staticmethod
    def for_beam_processor(comm):
        '''
        makes a candiate buffer for the beam processor.
        can call gather() to send to manager
        '''
        cands = MpiCandidateBuffer(MAX_NCAND_OUT)
        s_msg = [cands.cands,
                 MAX_NCAND_OUT,
                 cands.mpi_dtype]
        r_msg = [cands.cands, 0, cands.mpi_dtype]
        cands.comm = comm
        cands.r_msg = r_msg
        cands.s_msg = s_msg
        return cands




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
