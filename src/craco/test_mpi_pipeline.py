#!/usr/bin/env python
"""
Example pipeline using MPI util

Copyright (C) CSIRO 2020
"""

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from mpiutil import MpiPipeline

TAG_TRANSPOSE_DATA = 1
TAG_PREP_DATA = 2
TAG_CANDIDATES_LEVEL0 = 3
TAG_CANDIDATES_LEVEL1 = 4
TAG_CANDIDATES_LEVEL2 = 5
TAG_CANDIDATES_LEVEL3 = 6
TAG_CANDIDATES_LEVEL4 = 7
TAG_AVERAGE_DATA = 8
TAG_CUTOUT = 9
TAG_CALIBRATION_DATA = 10

class TestPipeline(MpiPipeline):
    def __init__(self, values):
        self.values = values
        self.beam_process(self.do_beam_processing)
        self.root_process(self.do_root_processing)
        
        self.values.dtype = np.complex64
        assert self.nchan % self.nfavg == 0, 'NFAVG must divide into nchan'
        self.nchan_out = self.values.nbeam*self.nchan_fout
        self.nant = self.values.nant
        self.nbl = self.nant*(self.nant-1)//2
        self.nhistory_blocks = 8


    def do_root_processing(self):
        pass

    def do_beam_processing(self):
        v = self.values
        self.input_data = np.zeros((v.nbeam, v.nchan_rx, self.nbl, v.nt), dtype=v.dtype)
        self.calibrated_data = np.zeros((v.nbeam*v.nchan_rx, self.nbl, v.nt), dtype=v.dtype)
        assert self.input_data.size == self.calibrated_data.size, 'Transpose must be regular, currently'
        self.cal_recv = np.zeros((v.nchan_out, self.nbl), dtype=v.dtype)
        self.cal_avg = np.zeros_like(self.cal_recv)
        self.calibration = np.ones_like(self.calibrated_data)
        self.sky = np.zeros_like(self.calibrated_data)

        self.history_buffer = np.zeros((self.nhistory_blocks, self.nchan_out, self.nbl, v.nt), dtype=v.dtype)

        for blk in range(self.nblocks):
            self.do_beam_processing_block(blk)

    def do_beam_processing_block(self, blkid):
        self.blkid = blkid
        self.read()
        self.prepare()
        self.transpose()
        self.transpose_calibration()
        self.calibrate()
        self.baseline2uv()
        self.search()
        self.classify_level0_candidate()
        self.cutout_l1()
        self.classify_level2_candidate()
        self.plan()


    def read(self):
        # Dummy read
        self.input_data.flat = np.arange(self.input_data.size)*blk
        

    def prepare(self):
        self.tavg = self.input_data.mean(axis=2)
        self.nchan_favg = v.nchan_rx // v.nfavg
        self.calibrated_data = self.input_data * self.calibration - self.sky
        self.favg = self.input_data.reshape(v.nbeam, self.chan_favg, v.nfavg, v.nt).mean(axis=2)

    def transpose(self):
        size = np.prod(self.favg.shape[1:])
        # These lines assume dtype has a size == 2*MPI_FLOAT
        assert self.favg.itemsize == 8
        assert self.calibrated_data.itemsize == 8
        send_msg = [self.favg, size*2, MPI.FLOAT]
        recv_msg = [self.calibrated_data, size*2, MPI.FLOAT]
        self.beam_comm.Alltoall(send_msg, recv_msg)

    def transpose_calibration(self):
        size = np.prod(self.tavg.shape[1:])
        send_msg = [self.tavg, size*2, MPI_FLOAT]
        recv_msg = [self.cal_recv, size*2, MPI_FLOAT]
        self.beam_comm.Alltoall(send_msg, recv_msg)

    def calibrate(self):
        # dummy for now - just average the data
        self.cal_avg += self.cal_recv
        blocks_per_calibration = 16
        do_calibration = self.blkid % blocks_per_calibration == (blocks_per_calibration - 1)

        if do_calibration:
            # In real life, we'd upate these with real numbers
            self.calibration *= 2
            self.sky += 1

    def baseline2uv(self):
        
                    
            

                
        
    def read_and_transpose(self):
        nbeam = self.nbeam
        nant = self.values.nant
        niter = self.values.niter
        
        nbl = nant*(nant - 1)//2
        nt = 256*6
        # Assume we start with all beams for 1 channel
        din = np.zeros((nbeam, nbl*nt), dtype=np.int32)
        print(f'Shape {din.shape} {din.nbytes} bytes')
        
        # outputs is all channels for 1 beam - but we assume 36 beams and 36 channels
        dout = np.zeros_like(din)
        size = nbl*nt # number of entries to zend
    
        send_msg = [din, size, MPI.INT]
        recv_msg = [dout, size, MPI.INT]

        comm = self.get_beam_communicator('read_and_transpose')
        pre_prank = self.get_beam_rank('prep')
        beamid = self.get_beam()

        print(f'Input for {rank} is {din}')
        for i in range(self.niter):
            start = time.time()
            din.flat = np.arange(din.size)*(beamid + self.nbeam*i) # synthetic data

            # transpose
            comm.Alltoall(send_msg, recv_msg)

            # send data to next step
            comm.Send(dout, dest=prep_rank, tag=TAG_TRANSPOSE_DATA)

            
    def prep(self):
        status = MPI.Status()
        cal_data = np.zeros((nbeam, nbl), dtype=np.int32)
        din = np.zeros((nbeam, nbl*nt), dtype=np.int32)
        dout = np.zeros_like(din)
        comm = MPI_COMM_WORLD
        process_rank = get_beam_rank('process')
        for i in range(self.niter):
            comm.probe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            if status.tag == TAG_TRANSPOSE_DATA:
                comm.Recv(din, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                do_prepare(din, dout, cal_data)
                comm.Send(din, dest=process_rank, tag=TAG_PREP_DATA)
            elif status.tag == TAG_CALIBRATION_DATA:
                comm.Recv(din, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            else:
                raise ValueError(f'Unknown tag:{status.tag}')

    def process(self):
        din = np.zeros((nbl, nchan, nt), dtype=np.int32)
        plan = None
        while True:
            comm.prob(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            if status.tag == TAG_PREP_DATA:
                comm.Recv(din, MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                do_processing(din, plan, candidates)
                comm.Send(candidates, candidate_rank, tag=TAG_CANDIDATES_LEVEL0)
            elif status.tag == TAG_PROCESS_LUTS:
                plan = comm.recv()
            else:
                raise ValueError(f'Unknown tag:{status.tag}')
        

    def run(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.numprocs = comm.Get_size()
        log.debug(f'My rank is {rank}/{numprocs}')

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--nbeam', type=int, help='Nbeam', default=36)
    parser.add_argument('--nchan_rx', type=int, help='Nchan', default=36*10)
    parser.add_argument('--nant', type=int, help='Nchan', default=30)
    parser.add_argument('--nt', type=int, help='NT', default=256)
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    p = TestPipeline(values)
    p.run()
    

if __name__ == '__main__':
    _main()
