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
from astropy import units as u
import datetime
import shutil
import time
import signal
import atexit
import numpy as np
from craco.prep_scan import touchfile,ScanPrep
from craft.cmdline import strrange
from craco.craco_run import auto_sched


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def mycaget(s):
    r = caget(s)
    if r is None:
        raise ValueError(f'PV {s} returned None')
    
    return r

scandir = None # Yuck. 
stopped = False
do_calibration = None


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--show-output', action='store_true', default=False, help='Show output on stdout rather than logging to logfile')
    parser.add_argument('-b','--beam', type=int, default=-1, help='Beam to download. -1 is all and default and enables tscrunch')
    parser.add_argument('--scan-minutes', type=float, help='Number of minutes to record for', default=15)
    parser.add_argument('--pol-sum', help='Sum pol mode', action='store_true', dest='pol_sum', default=True)
    parser.add_argument('--dual-pol', help='Dual pol mode', action='store_false', dest='pol_sum', default=False)
    parser.add_argument('-a','--card', help='Cards to download', default='1-12')
    parser.add_argument('--block', help='Blocks to download', default='5-7')
    parser.add_argument('--max-ncards', help='Number of cards to download', type=int, default=30)
    parser.add_argument('--transpose', help='Do the transpose in real time', action='store_true', default=False)    
    parser.add_argument('--metadata', help='Prep scan with this metadata file')
    parser.add_argument('--flag-ants', help='Antennas to flag', default='31-36', type=strrange)
    parser.add_argument('--search-beams', help='Beams to search')
    parser.add_argument('--phase-center-filterbank', help='Phase center filterbank')
    parser.add_argument('--trigger-threshold', help='Triggerr threshold', type=float, default=10)
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
    
    scandir = os.environ['SCAN_DIR']
    prep = ScanPrep.load(scandir)
    scanid = prep.scan_id
    sbid = prep.sbid
    target = prep.targname

    calfinder = auto_sched.CalFinder(sbid)
    calpath = calfinder.get_cal_path() # None if nothign available.
    global do_calibration
    do_calibration = calpath is None # do a calibration scan if no calibration available.

    # make soft link to calibration path for andy
    if calpath is not None:
        cal_link = os.path.join(scandir,'../../../cal')
        if not os.path.exists(cal_link):
            os.symlink(calpath, cal_link)


    os.chdir(scandir)
    touchfile('SCAN_START', directory=scandir, check_exists=False)        
    target_file = os.path.join(scandir, 'ccap.fits')
    log.info(f'Saving scan SB{sbid} scanid={scanid} target={target} to {scandir}')

    #cmdname='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpipipeline.sh'
    #hostfile='/data/seren-01/fast/ban115/build/craco-python/mpitests/mpi_seren.txt'
    hostfile = os.environ['HOSTFILE']
    shutil.copy(hostfile, scandir)
    beam = values.beam # all beams, tscrucnh


    if values.transpose:
        cmdname = 'mpipipeline.sh'
    else:
        cmdname = 'mpicardcap.sh'

    #beam = 0 # given beam no tscrunch
    if values.pol_sum:
        pol = '--pol-sum'
    else:
        pol = '--dual-pol'

    if values.phase_center_filterbank:
        pcb = f'--phase-center-filterbank {values.phase_center_filterbank}'
    else:
        pcb = ''

    search_vis_tscrunch = '--vis-tscrunch 4' # if we're conservative
    #search_vis_tscrunch = '--vis-tscrunch 2' # going to guns!
    # tscrunch scrunches the packets
    calibration = '' if calpath is None else f'--calibration {calpath}'
    spi = '--samples-per-integration 64' if beam == -1 else '--samples-per-integration 32' # spi64 for all beams, or spi32 for single beam
    tscrunch = '--tscrunch 32' if beam == -1 else '' # tscrunch packets for all beams. Otherwise leave them alone. But tscrunc is only for non transposing
    vis_tscrunch = '--vis-tscrunch 32' if do_calibration else search_vis_tscrunch # when calibrating make it 110ms in UVFITS file

    card  = f'--card {values.card}'
    block = f'--block {values.block}'
    fpga = ''
    # 30 cards is about the limit for cardcap
    max_ncards = f'--max-ncards {values.max_ncards}'


    if do_calibration:
        scan_nminutes = 2
    else:
        scan_nminutes = values.scan_minutes

    nmsg = int(scan_nminutes*60/.11) #  Number of blocks to record for
    num_msgs = f'-N {nmsg}'
        
    num_cmsgs = '--num-cmsgs 1'
    num_blocks = '--num-blks 16'
    fcm = '--fcm /home/ban115/20220714.fcm'
    metafile = '--metadata {values.metadata}' if values.metadata else ''
    ndm = '--ndm 256'

    if values.search_beams and not do_calibration:
        search_beams = f'--search-beams {values.search_beams}'
    else:
        search_beams = ''   


    valid_ants = set(prep.valid_ant_numbers) # 1 based antenna numbers to include

    # Antenna handling for 36 antennas is a whole massive issue. WE're not going to handle that now. 
    # we'll just assume we have 30 anntennas - there are all sorts of bugs in mpipipeline that cant handle sending through flagged ants 31-36
    all_ants = set(np.arange(30) + 1)
    flagged_ants = all_ants - valid_ants # 1 based antenna number to not include
    flagged_ants = flagged_ants.union(set(values.flag_ants)) # also flag antennas from the cmdline
    flagged_ants = flagged_ants - set(np.arange(6) + 31)
    flag_ant_str = ','.join(sorted(list(map(str, flagged_ants))))
 

    if flag_ant_str:
        antflag = f'--flag-ants {flag_ant_str}'
    else:
        antflag = ''    

    # for mpicardcap
    if values.transpose:
        cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} {pol} {spi} {card} {fpga} {block} {max_ncards} {pcb} --outdir {scandir} {fcm} --transpose-nmsg=2 --save-uvfits-beams 0-35 {vis_tscrunch} {metafile} {antflag} {search_beams} {calibration} {ndm} --trigger-threshold {values.trigger_threshold}'
    else:
        cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} -f {target_file} {pol} {tscrunch} {spi} {beam} {card} {fpga} {block} {max_ncards} --devices mlx5_0,mlx5_2 {antflag}'

    log.info(f'Running command {cmd}')

    atexit.register(exit_function)
    # subprocess.run doesn't work - if you kill with sigterm you get 'terminated' written to stdout - no exception, exit function isn't run.
    #
    if values.show_output:
        logfile = None
    else:
        logfile = open(os.path.join(scandir, 'run.log'), 'w')

    env = os.environ.copy()
    proc = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.STDOUT, env=env)
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
    global stopped
    global do_calibration
    if not stopped:
        stopped = True
        log.info('Stopping CRACO in exit_function. Do cal? %s', do_calibration)
        caput('ak:cracoStop', 1)
        scandir = os.environ['SCAN_DIR']
        touchfile('SCAN_STOP', directory=scandir, check_exists=False)
        if do_calibration:
            log.info('Queing calibration')
            auto_sched.queue_calibration(scandir)

if __name__ == '__main__':
    _main()
