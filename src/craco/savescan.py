#!/usr/bin/env python
"""
Saves a scan. Run by obsnam

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
from epics import PV,caget
import subprocess
import datetime
import shutil

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def mycaget(s):
    r = caget(s)
    if r is None:
        raise ValueError(f'PV {s} returned None')
    
    return r
    

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    scanid = mycaget('ak:md2:scanId_O')
    sbid = mycaget('ak:md2:schedulingblockId_O')
    target = mycaget('ak:md2:targetName_O')
    bigdir = os.environ['DATA_DIR_BIG']
    fastdir = os.environ['DATA_DIR_FAST']
    now = datetime.datetime.utcnow()
    nowstr = now.strftime('%Y%m%d%H%M%S')
    scandir = os.path.join(bigdir, f'SB{sbid:06}', target.replace(' ','_'), f'SCAN{scanid:02d}', nowstr)
    os.makedirs(scandir)
    os.chdir(scandir)
    target_file = os.path.join(scandir, 'ccap.fits')
    log.info(f'Saving scan SB{sbid} scanid={scanid} target={target} to {scandir}')
    cmdname='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpicardcap.sh'
    hostfile='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpi_seren.txt'
    shutil.copy(hostfile, scandir)
    
    cmd = f'{cmdname} mpi_seren.txt -e --prefix ak --num-msgs 1000000 -f {target_file} --pol-sum --tscrunch 64 --samples-per-integration 32 -a 1-12 -b 2-4'

    log.info(f'Running command {cmd}')
    try:
        completion = subprocess.run(cmd, shell=True)
        log.info(f'Command {cmd} completed with return code {completion.returncode}')
    except KeyboardInterrupt:
        log.info('Command {cmd} interrupted by keyboard exception')



    # TODO: postprocessing
    


    

if __name__ == '__main__':
    _main()
