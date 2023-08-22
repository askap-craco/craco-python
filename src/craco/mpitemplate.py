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
import glob

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import mpi4py.util.dtlib


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--dump-rankfile', help='Rankfile name to dump')
    parser.add_argument('--sbdir', help='Schedblcok directory')
    
    #ls /data/seren-??/big/craco/SB051999/scans/00/20230821121442/cal/*.uvfits

    parser.set_defaults(verbose=False)
    values = parser.parse_args()
        
    if values.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger()
    logger.addHandler(mpiutil.make_log_handler(comm))
    logger.setLevel(level)
    comm.Barrier()
    log.info('Hello world! Running with %s', values)

    if values.dump_rankfile is not None:
        total_glob = os.path.join('/data/seren-??/big/craco/', values.sbdir, 'cal/*.uvfits')
        all_files = glob.glob(total_glob)
        with open(values.dump_rankfile, 'w') as fout:
            for f in sorted(all_files):
                bits = f.split('/')
                host = bits[2]
                fname = bits[-1]
                beamno = int(fname.split('.')[0].replace('b',''))
                print(host, beamno)
                #rank 0=seren-01 slot=1:0 # Block 2 card 1 fpga 1
                fout.write(f'rank {beamno}={host} slot=1:0-5 # Beam {beamno}\n')

        sys.exit(0)

    myrank = comm.Get_rank()
    if rank == 0:
        log.info('WOW Im the master rannkin the whole shebanin********')

    beamno = rank

    targetfile = os.path.join('/data/big/craco/', values.sbdir, f'cal/b{beamno:02d}.uvfits')
    log.info('Target file is %s = %d bytes', targetfile, os.path.getsize(targetfile))
    comm.Barrier()


    

if __name__ == '__main__':
    _main()
