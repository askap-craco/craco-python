#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
from epics import PV, caget,caput
from subprocess import Popen, TimeoutExpired
import time
import signal
import atexit
import re
import datetime
import numpy as np
from craft.cmdline import strrange

from craco.prep_scan import ScanPrep, make_scan_directories
from craco.scan_manager import ScanManager,NANT

from craco.askap.craft.obsman.sbstatemonitor import SBStateSubscriber

# pylint: disable-msg=E0611
# need to import iceutils before interfaces
from askap.iceutils import get_service_object
import askap.interfaces as iceint
from askap.interfaces.schedblock import ObsState

from craco.askap.craft.obsman.metadatasaver import MetadataSaver
from craco.askap.craft.obsman.metadatasubscriber import metadata_to_dict
from askap.parset import ParameterSet

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Obsman:
    def __init__(self, values):
        self.process = None        
        self.cmd = ' '.join(values.cmd)
        self.values = values
        self.doquit = False
        self.curr_scanid = None


        # Bit of a race condition here, but we'll do it
        # add_callback might have beaten us to it,
        #self.scan_changed(self.target_pv.pvname, self.target_pv.get())
        #self.callback_id = self.target_pv.add_callback(self.scan_changed)
        #self.scan_changed(self.scan_pv.pvname, self.scan_pv.get())
        #self.callback_id = self.scan_pv.add_callback(self.scan_changed)

        atexit.register(self.atexit_handler)
        signal.signal(signal.SIGTERM, self.sigterm_handler)
        signal.signal(signal.SIGHUP, self.sighup_handler)
        signal.signal(signal.SIGINT, self.sigint_handler)

    def atexit_handler(self):
        log.info('Running atexit function')
        self.terminate_process()

    def sigterm_handler(self, signal, frame):
        log.info('Got sigterm')
        self.shutdown()

    def sighup_handler(self, signal, frame):
        log.info('Got sighup')
        self.restart_process()

    def sigint_handler(self, signal, frame):
        log.info('Got sigint')
        self.shutdown()

    def restart_process(self):
        running = self.is_running
        self.terminate_process()
        # restart
        
        if running:
            pass # don't restart yet - I think we need to load up a new scan info so we don't end up restaring with the same metadadta?
            self.start_process(self.curr_scan_info)

    @property
    def is_running(self):
        return self.process is not None
        
    def scan_changed(self, scan_info):
        sbid = scan_info.sbid
        new_scanid = scan_info.scan_id
        target = scan_info.targname
        
        match = None
        if self.values.target_regex is not None:
            match = re.search(self.values.target_regex, target)
            target_ok = match is not None
        else:
            target_ok = True

        craco_enabled = caget('ak:enableCraco') == 1
        zoomval = caget('ak:S_zoom:val')
        standard_zoom = zoomval == 0

        log.info(f'Scan_changed newscanID={new_scanid} currscan={self.curr_scanid} SB{sbid} target={target} OK?={target_ok} CRACO enabled?={craco_enabled} zoomval = {zoomval} zoom OK? = {standard_zoom}')
        if new_scanid == -2 or new_scanid is None: # it's closing - sometimes glitches
            self.terminate_process()
        elif new_scanid == -1: # it's getting ready, do nothign
            pass
        elif new_scanid == self.curr_scanid: # avoid race condition
            pass
        elif target_ok and craco_enabled and standard_zoom: # new valid scan number with new scan ID and valid zoom
            assert new_scanid >= 0 and new_scanid != self.curr_scanid
            self.start_process(scan_info)
        else:
            log.info('Passing on %s as doesnt match regex %s or craco enabled=%s', target, self.values.target_regex, craco_enabled)

    def start_process(self, scan_info):
        # terminate process to start with
        self.terminate_process()

        assert self.curr_scanid is None
        assert self.process is None

        sbid = scan_info.sbid
        new_scanid = scan_info.scan_id
        target = scan_info.targname
        outdir = scan_info.outdir
     
        self.curr_scanid = new_scanid
            
        # upate environment
        env = {'SB_ID': str(sbid),
               'SCAN_ID': str(new_scanid),
               'TARGET': target,
               'SCAN_DIR': outdir
        }
        cmd = self.cmd.format(**env)
        myenv = os.environ.copy()
        myenv.update(env)

        # start new process group and use it to kill all subprocesses. on exit
        # I love you so much Alexandra
        # https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
        log.debug('Running command  "%s" with extra environ %s', cmd, env)
        self.process = Popen(cmd.split(), shell=False, start_new_session=True, env=myenv)
        pgid = os.getpgid(self.process.pid)
        self.scan_info = scan_info
        
        self.start_time = datetime.datetime.now()
        self.sb_id = sbid
        self.scan_id = new_scanid
        self.target = target
        self.outdir = outdir
        self.curr_scan_info = scan_info
        log.info(f'Started process {cmd} with PID={self.process.pid} PGID={pgid} retcode={self.process.returncode} SBID=%s SCAN_ID=%s TARGET=%s SCAN_DIR=%s', sbid, new_scanid, target, outdir)
        
    def terminate_process(self):
        proc = self.process
        if self.process is None:
            log.info('terminate process called but process is none')
            return

        exit_code = self.process.poll()
        log.info('In terminate_process(). Poll returns %s', exit_code)
       
        if exit_code is None: #process is still running
            try:
               pgid = os.getpgid(proc.pid) # get process group ID - which we got our own when we started the session
               timeout = self.values.death_timeout
               log.info(f'sending SIGINT processes from PID={proc.pid} PGID={pgid}and waiting {timeout} seconds')
               proc.send_signal(signal.SIGINT)
               os.killpg(pgid, signal.SIGINT)
               try:
                   proc.wait(timeout)
               except TimeoutExpired:
                   log.warning('Process didnt complete after terminate. Doing kill')
                   os.killpg(pgid, signal.SIGKILL)
                   
            except ProcessLookupError:# os.getpgid throws this if "your men are already dead" - Agent Smith
                log.info('Process group for %s does not exist. Didnt kill but should cleaup automatically', proc.pid)
                    
            proc.terminate()

        # by this point we've sent all the signals so, we just wait
        exit_code = self.process.poll()
        if exit_code is None:
            log.info('Now doing eternal wait on process. Exit code is %s', exit_code)
            proc.wait()
            
        retcode = proc.returncode
        log.info('Process dead with return code %s. Stopping packets just in case', retcode)
        
        caput('ak:cracoStop', 1)
        
        # assert retcode is not None
        self.process = None
        self.curr_scanid = None
        log.info('CRACO stopped. process is None. Scanid is None')

    @property
    def is_running(self):
        return self.process is not None    

    def poll_process(self):
        '''
        Called to check proces is running
        If it quits by itself, it's cleaned up and restarted
        '''
        if self.process is not None:
            retcode = self.process.poll()
            now = datetime.datetime.now()
            minutes = (now - self.start_time).total_seconds()/60
            log.debug('Process pid=%s running with return code %s for %0.1f minutes %s', self.process.pid, retcode, minutes, self.outdir)
            if retcode is not None or minutes > self.values.timeout:
                log.info('Process DIED UNPROVOKED with return code %s or timeout with %0.1f > %0.1f. Cleaning up and restarting %s', retcode, minutes, self.values.timeout, self.outdir)
                #self.restart_process()
                # were going to terminate and then get it to make a new scan
                self.terminate_process()
        
        return self.is_running

    def shutdown(self):
        log.info('Shutting down process')
        #self.scan_pv.remove_callback(self.callback_id)
        self.terminate_process()
        sys.exit()


class EpicsObsmanDriver:
    def __init__(self, obsman):
        self.scan_pv = PV('ak:md2:scanId_O')
        self.sbid_pv = PV('ak:md2:schedulingblockId_O')
        self.target_pv = PV('ak:md2:targetName_O')
        self.obsman = obsman
        
    def wait(self):
        try:
            scanid = self.scan_pv.get()
            sbid = self.sbid_pv.get()
            target = self.target_pv.get() # this doesnt refresh for some reason
            # initial setup
            info = ScanPrep(target, sbid, scanid)
            self.obsman.scan_changed(info)
            while True:
                time.sleep(1)
                self.obsman.poll_process()
                new_scanid = self.scan_pv.get()
                if new_scanid != scanid:
                    sbid = self.sbid_pv.get()
                    target = self.target_pv.get() # this doesnt refresh for some reason
                    info = ScanPrep(target, sbid, new_scanid)
                    self.obsman.scan_changed(info)
                    scanid = new_scanid
        except KeyboardInterrupt:
            log.info('Ctrl-C detected')
        except:
            log.exception('Failiure polling process')
        finally:
            self.obsman.shutdown()



def get_ant_numbers_from_obs_variables(obs_variables):
    # sb_ants is a list like ['ant1','ant2',...,'ant36']
    sb_ants = obs_variables['schedblock.antennas']

    # convert to boolean mask
    ant_numbers = np.array(list(map(lambda x: int(x.replace('ant','')), sb_ants)))
    return ant_numbers


# Wow  Subclassing MetadataSaver never received any metdata
# so weird. Addit and a listener and it works fine.
#class MetadataObsmanDriver(MetadataSaver):
class MetadataObsmanDriver:
    def __init__(self, comm, obsman):
        self.obsman = obsman
        self.sbid = None
        self.scan_running = False

        values = self.obsman.values        
        self.ignore_ants_1based = np.array(values.ignore_ants)
        self.scan_manager = None
        
        self.sb_service = get_service_object(comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)


    def changed(self, sbid, state, updated, old_state, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        log.info('SB STATE CHANGED sbid=%s state=%s updated=%s old_state=%s', sbid, state, updated, old_state)

        if state == ObsState.EXECUTING:
            self.sbid = sbid
            # pick up antenna list from observation variables

        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None
            self.scan_manager = None

    def publish(self, pub_data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320

        '''
        if self.sbid is None:            
            return

        mgr = self.get_scan_manager()
        if mgr is None:
            return
                            
        d = metadata_to_dict(pub_data, self.sbid)

        next_scan_running = mgr.push_data(d)
        self.scan_running = self.obsman.poll_process()
        if self.scan_running:
            if next_scan_running: # continue running scan
                pass                
            else: # metadata says to stop running scan
                self.obsman.terminate_process()
        else:
            if next_scan_running: # start new scan
                info = ScanPrep.create_from_metafile(mgr.scan_metadata, self.ant_numbers)
                self.obsman.scan_changed(info)
            else:
                pass # continue not running a scan
            
        self.obsman.poll_process()
        self.scan_running = next_scan_running and self.obsman.is_running

    def close(self):
        self.obsman.terminate_process()

    def get_scan_manager(self):
        sbid = self.sbid
        log.debug('In get scan maanger %s sbid=%d', self.scan_manager, sbid)
        if self.scan_manager is not None:
            return self.scan_manager
        
        for retry in range(3):               
            self.obs_variables = ParameterSet(self.sb_service.getObsVariables(sbid, ''))
            # for some reason obs variables are not updated when the thing starts. Grrr.
            if 'schedblock.antennas' in self.obs_variables:
                break
            log.warning('No antennas for schedblock %d. sleeping.', sbid)
            time.sleep(1)

        if 'schedblock.antennas' in self.obs_variables:           
            orig_ant_numbers = get_ant_numbers_from_obs_variables(self.obs_variables)
            self.ant_numbers = np.setdiff1d(orig_ant_numbers, self.ignore_ants_1based)
            
            self.scan_manager = ScanManager(self.ant_numbers, frac_onsource=self.obsman.values.frac_onsource )
            log.info('%d/%d active antennas %s', len(self.ant_numbers), NANT, ','.join(self.ant_numbers.astype('str')))
        else:
            log.warning('No antennas for schedblock %d. Waiting for next smetadata.', sbid)
            self.scan_manager = None

        log.info('END get scan maanger %s sbid=%d', self.scan_manager, sbid)
        return self.scan_manager


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--death-timeout', type=int, help='Timeout to wait after sending signal before killing process', default=10)
    parser.add_argument('--timeout', type=float, help='Number of minutes to wait before killing process', default=15)
    parser.add_argument('-R','--target-regex', help='Regex to apply to target name. If match then we start a scan')
    parser.add_argument(dest='cmd', nargs='+')
    parser.add_argument('--force-start', action='store_true', help='Start even if metadata says not to. Useful for testing')
    parser.add_argument('--driver', choices=('meta','epics'), default='meta', help='DRive with epics or metadata')
    parser.add_argument('--frac-onsource', default=0.9, type=float, help='Fraction of antennas that need to be on source to start a scan')
    parser.add_argument('--ignore-ants', type=strrange, default='', help='List of antennas to remove from the list of available antenans')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(processName)s (%(process)d) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(filename)s.%(funcName)s (%(process)d) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    obs = Obsman(values)
    if values.driver == 'meta':
        import Ice
        # TODO: Wrap this nicely.
        comm = Ice.initialize(sys.argv)
        d = MetadataObsmanDriver(comm, obs)
        saver = MetadataSaver(comm, d)
        try:
            comm.waitForShutdown()
        finally:
            log.info('Desctroying comm')
            comm.destroy()
    else:
        d = EpicsObsmanDriver(obs)
        d.wait()
    

if __name__ == '__main__':
    _main()
