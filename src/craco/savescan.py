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
from craco import summarise_cands, summarise_scan, scan_archiver

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
obsparams = None

def get_param_with_default(obsparams, key, default_value):
    '''
    Gets key from obsparams with get_value. If value is not specified or equals
    '', 'None' or 'default' returns the default value. Otherwise returns the default value
    '''
    v = obsparams.get_value(key, '')
    if v == '' or v == 'None' or v == 'default':
        v = default_value
    
    return v



def calc_inttime_tscrunch(inttime_exp:int):
    '''
    For a given integration time of 864us*(2**inttime_exp) calculate the samples-per-integration parameter
    and remaining tscrunch parameter to give you the desired inttime
    samples per integration can be 16 (0.864ms), 32 (1.7ms) or 64 (3.4ms). Anything leftover is tscrunched
    '''
    assert isinstance(inttime_exp, int)
    assert 0 <= inttime_exp, f'Invalid interagration time exponent {inttime_exp}'
    inttime = 1<<inttime_exp # multiples of 864us
    if inttime == 1: # 864us
        spi = 16        
    elif inttime == 2: # 1.7ms
        spi = 32        
    elif inttime == 4: # 3.4ms
        spi = 64        
    else:
        spi = 64

    tscrunch = (inttime)//(spi//16)
    assert tscrunch >= 1
    return (spi, tscrunch)


class MyParams:
    def __init__(self, values, obsparams):
        self.values = values
        self.obsparams = obsparams

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--show-output', action='store_true', default=False, help='Show output on stdout rather than logging to logfile')
    parser.add_argument('-b','--beam', type=int, default=-1, help='Beam to download. -1 is all and default and enables tscrunch')
    parser.add_argument('--scan-minutes', type=float, help='Number of minutes to record for', default=15)
    parser.add_argument('--calibration-scan-minutes', default=15.0, type=float, help='Length of scan if we need to do a calibartion')
    parser.add_argument('--pol-sum', help='Sum pol mode', action='store_true', dest='pol_sum', default=True)
    parser.add_argument('--dual-pol', help='Dual pol mode', action='store_false', dest='pol_sum', default=False)
    parser.add_argument('-a','--card', help='Cards to download', default='1-12')
    parser.add_argument('--block', help='Blocks to download', default='2-7')
    parser.add_argument('--max-ncards', help='Number of cards to download', type=int, default=72)
    parser.add_argument('--transpose', help='Do the transpose in real time', action='store_true', default=True)  
    parser.add_argument('--no-transpose', dest='transpose', action='store_false', help='Dont do transpose')  
    parser.add_argument('--metadata', help='Prep scan with this metadata file')
    parser.add_argument('--flag-ants', help='Antennas to flag', default='31-36', type=strrange)
    parser.add_argument('--search-beams', help='Beams to search')
    parser.add_argument('--phase-center-filterbank', help='Phase center filterbank')
    parser.add_argument('--trigger-threshold', help='Trigger threshold', type=float, default=10)
    parser.add_argument('--update-uv-blocks', default=6, type=int, help='Update uv blocks')
    parser.add_argument('--int-time-exp', type=int, default=4, help='Integration time as 864us*2**X')
    parser.add_argument('--ndm', type=int, default=256, help='Number of DM trials')
    parser.add_argument('--force-calibration', action='store_true', default=False, help='Make it do calibration even if one exists')
    parser.add_argument('--no-save-uvfits', action='store_false', dest='save_uvfits', default=True, help='Dont save uvfits to disk')
    parser.add_argument(dest='extra', help='Extra arguments', nargs='*')
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
    global obsparams
    obsparams = prep.load_parset('params.parset')

    calfinder = auto_sched.CalFinder(sbid)
    calpath = calfinder.get_cal_path() # None if nothign available.
    global do_calibration
    do_calibration = calpath is None or values.force_calibration# do a calibration scan if no calibration available.

    # make soft link to calibration path for andy
    if calpath is not None:
        cal_link = os.path.join(scandir,'../../../cal')
        if not os.path.exists(cal_link):
            os.symlink(calpath, cal_link)


    os.chdir(scandir)
    touchfile('SCAN_START', directory=scandir, check_exists=False)        
    target_file = os.path.join(scandir, 'ccap.fits')
    log.info(f'Saving scan SB{sbid} scanid={scanid} target={target} to {scandir} calibration={calpath} and extra args {values.extra}')

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


    calibration = '' if do_calibration else f'--calibration {calpath}'
    uvfits_required = 'craco.uvfits.int_time_exp' in obsparams.keys()
    int_time_exp = int(get_param_with_default(obsparams, 'craco.uvfits.int_time_exp', values.int_time_exp))
    
    if do_calibration:
        int_time_exp = 7 # Do more averaging during calbration 110 ms

    int_time = 0.864*2**(int_time_exp)
    spi_val, tscrunch_val = calc_inttime_tscrunch(int_time_exp)
    log.info('Using integration time=%sms spi=%s tscrunch=%s', int_time, spi_val, tscrunch_val)
   
    spi = f'--samples-per-integration {spi_val}'
    vis_tscrunch = f'--vis-tscrunch {tscrunch_val}'

    # only used if not transposing
    tscrunch = '--tscrunch 32' if beam == -1 else '' # tscrunch packets for all beams. Otherwise leave them alone. But tscrunc is only for non transposing

    card  = f'--card {values.card}'
    block = f'--block {values.block}'
    fpga = ''
    # 30 cards is about the limit for cardcap
    max_ncards = f'--max-ncards {values.max_ncards}'

    if do_calibration:
        scan_nminutes = values.calibration_scan_minutes
    else:
        scan_nminutes = float(get_param_with_default(obsparams, 'craco.scan_minutes', values.scan_minutes))

    nmsg = int(scan_nminutes*60/.11) #  Number of blocks to record for
    num_msgs = f'-N {nmsg}'
        
    num_cmsgs = '--num-cmsgs 1'
    num_blocks = '--num-blks 16'
    fcm = '--fcm /home/ban115/20220714.fcm'
    metafile = '--metadata {values.metadata}' if values.metadata else ''
    ndm_val = int(values.ndm)
    ndm = f'--ndm {ndm_val}'

    if values.search_beams and not do_calibration:
        search_beams = f'--search-beams {values.search_beams}'
    else:
        search_beams = ''

    if values.save_uvfits or do_calibration or uvfits_required:
        save_uvfits = f'--save-uvfits-beams 0-35'
    else:
        save_uvfits = ''


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

    freqfile = prep.fixed_flag_file
    freqfile = os.path.abspath(freqfile).replace('/data/craco/craco', '/CRACO/DATA_00/craco')
    assert os.path.exists(freqfile), f'Freqfile doesnt exist {freqfile}'
    flag_file = f'--flag-frequency-file {freqfile}'

    update_uv_blocks = f'--update-uv-blocks {values.update_uv_blocks}'
    extra = ' '.join(values.extra)

    # for mpicardcap
    if values.transpose:
        cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} {pol} {spi} {card} {fpga} {block} {max_ncards} {pcb} --outdir {scandir} {fcm} --transpose-nmsg=2  {save_uvfits} {vis_tscrunch} {metafile} {antflag} {search_beams} {calibration} {ndm} --trigger-threshold {values.trigger_threshold} {update_uv_blocks} {flag_file} {extra}'
    else:
        cmd = f'{cmdname} {num_cmsgs} {num_blocks} {num_msgs} -f {target_file} {pol} {tscrunch} {spi} {beam} {card} {fpga} {block} {max_ncards} --devices mlx5_0,mlx5_2 {antflag} {extra}'

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
    global obsparams
    if not stopped:
        stopped = True
        log.info('Stopping CRACO in exit_function. Do cal? %s', do_calibration)
        caput('ak:cracoStop', 1)
        scandir = os.environ['SCAN_DIR']
        touchfile('SCAN_STOP', directory=scandir, check_exists=False)
        if do_calibration:
            log.info('Queing calibration')
            auto_sched.queue_calibration(scandir)
        summarise_cands.run_with_tsp()
        summarise_scan.run_with_tsp()
        auto_sched.run_post_cand_with_tsp()
        
        archive_location  = get_param_with_default(obsparams, 'craco.archive.location', '')
        log.info('craco.archive.location location is %s', archive_location)
        if archive_location != '':
            scan_archiver.run_with_tsp(archive_location)

if __name__ == '__main__':
    _main()
