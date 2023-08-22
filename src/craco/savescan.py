#!/usr/bin/env python
"""
Saves a scan. Run by obman

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
from epics import PV,caget,caput
import subprocess
from subprocess import Popen, TimeoutExpired
import datetime
import shutil
import time
import signal
import atexit

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
    parser.add_argument('--show-output', action='store_true', default=False, help='Show output on stdout rather than logging to logfile')
    parser.add_argument('-b','--beam', type=int, default=-1, help='Beam to download. -1 is all and default and enables tscrunch')
    parser.add_argument('--scan-minutes', type=float, help='Number of minutes to record for', default=15)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    lformat='%(asctime)s %(levelname)-8s %(filename)s.%(funcName)s (%(process)d) %(message)s'
    if values.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format=lformat,
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=lformat,
            datefmt='%Y-%m-%d %H:%M:%S')

    scanid = mycaget('ak:md2:scanId_O')
    sbid = mycaget('ak:md2:schedulingblockId_O')
    target = mycaget('ak:md2:targetName_O')
    bigdir = os.environ['CRACO_DATA']
    now = datetime.datetime.utcnow()
    nowstr = now.strftime('%Y%m%d%H%M%S')
    scandir = os.path.join(bigdir, f'SB{sbid:06}', 'scans', f'{scanid:02d}', nowstr)
    targetdir = os.path.join(bigdir, f'SB{sbid:06}', 'targets', target.replace(' ','_'))
    targetlink = os.path.join(targetdir, nowstr)
    os.makedirs(scandir)
    os.chdir(scandir)
    os.makedirs(targetdir, exist_ok=True)
    os.symlink(scandir, targetlink)
    target_file = os.path.join(scandir, 'ccap.fits')
    log.info(f'Saving scan SB{sbid} scanid={scanid} target={target} to {scandir}')
    cmdname='mpicardcap.sh'
    #cmdname='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpipipeline.sh'
    hostfile='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpi_seren.txt'
    shutil.copy(hostfile, scandir)
    #pol='--pol-sum'

    beam = -1 # all beams, tscrucnh
    #beam = 0 # given beam no tscrunch

    if beam == -1: # all beams
        beam = '' # all beams
        pol = '--dual-pol'
        tscrunch='--tscrunch 32'
        spi='--samples-per-integration 64'
    else:
        beam = f'--beam {beam}' # single beam
        pol = '--pol-sum'
        tscrunch = ''
        spi = '--samples-per-integration 32'

    card  = '-a 1-12'
    block = '-b 5-7'
    fpga = ''
    fpga_mask = ''

    # 30 cards is about the limit for cardcap
    max_ncards = '--max-ncards 30'

    if values.scan_minutes is not None:
        scan_nminutes = values.scan_minutes
        nmsg = int(scan_nminutes*60/.11) #  Number of blocks to record for
        num_msgs = f'-N {nmsg}'
    else:
        num_msgs = ''
        
    num_cmsgs = '--num-cmsgs 1'
    num_blocks = '--num-blks 16'

    # for mpicardcap
    cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} -f {target_file} {pol} {tscrunch} {spi} {beam} {card} {fpga} {block} {max_ncards}'

    # for mpipipeline
    #cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} {pol} {tscrunch} {spi} {beam} {card} {fpga} {block} {max_ncards} --outdir {scandir}'


    log.info(f'Running command {cmd}')

    atexit.register(exit_function)
    # subprocess.run doesn't work - if you kill with sigterm you get 'terminated' written to stdout - no exception, exit function isn't run.
    #
    if values.show_output:
        logfile = None
    else:
        logfile = open(os.path.join(scandir, 'run.log'), 'w')
    
    proc = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.STDOUT)
    finish = False

    def signal_handler(sig, frame):
        log.info(f'{__file__} got signal {sig}. Killing {proc.pid}')
        finish = True
        proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not finish:
            time.sleep(1)
            exit_code = proc.poll()
            currsig  = signal.getsignal(signal.SIGINT)
            log.debug('Process %s returncode is %s. Current signal is %s', proc.pid, proc.returncode, currsig)
            if proc.poll() is not None:
                log.info('Process %s terminated with return code %s', proc.pid, proc.returncode)
                finish = True
            
    except KeyboardInterrupt:
        log.info('Savescan received KeyboardInterrupt/SIGINT - terminating subprocess')
    finally:
        log.info('Terminating process')
        proc.terminate()
        proc.wait()
        if logfile is not None:
            logfile.close()

    log.info(f'Process completed with returncode {proc.returncode}')
    exit_function()
    retcode = 0
    log.info('Savescan complete. Returning %s', retcode)
    sys.exit(retcode)


def exit_function():
    log.info('Stopping CRACO in exit_function')
    caput('ak:cracoStop', 1)

if __name__ == '__main__':
    _main()
