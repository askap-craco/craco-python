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
from craco.prep_scan import ScanPrep, make_scan_directories
from craco.scan_manager import ScanManager
from craco.askap.craft.obsman.metadatasaver import MetadataSaver
from craco.askap.craft.obsman.metadatasubscriber import metadata_to_dict
from askap.interfaces.schedblock import ObsState


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Obsman:
    def __init__(self, values):
        self.process = None        
        self.cmd = values.cmd
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
        scanid = self.curr_scanid
        self.terminate_process()
        # restart
        if scanid is not None:
            self.start_process(scanid)
        
    def scan_changed(self, sbid, new_scanid, target, metadata=None):
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
            self.start_process(new_scanid)
        else:
            log.info('Passing on %s as doesnt match regex %s or craco enabled=%s', target, self.values.target_regex, craco_enabled)

    def start_process(self, sbid, new_scanid, target, metadata=None):
        assert new_scanid is not None
        # terminate process to start with
        self.terminate_process()

        assert self.curr_scanid is None
        assert self.process is None
        self.curr_scanid = new_scanid

        
        outdir = make_scan_directories(sbid, new_scanid, target)
        if metadata is not None:
            # creates directories, copies in metafile, adds some extra goodies.
            # runs calc for the hell of it.            
            prep = ScanPrep.create_from_metafile_and_fcm(metadata)
            
        # upate environment
        env = {'SB_ID': sbid,
               'SCAN_ID': new_scanid,
               'TARGET': target,
               'SCAN_DIR': outdir
        }
        cmd = self.cmd.format(**env)
        myenv = os.envrion.copy()
        myenv.extend(env)

        # start new process group and use it to kill all subprocesses. on exit
        # I love you so much Alexandra
        # https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/

        self.process = Popen(cmd, shell=False, start_new_session=True, env=myenv)
        pgid = os.getpgid(self.process.pid)
        self.start_time = datetime.datetime.now()
        log.info(f'Started process {self.cmd} with PID={self.process.pid} PGID={pgid} retcode={self.process.returncode}')
        
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
        

    def poll_process(self):
        '''
        Called to check proces is running
        If it quits by itself, it's cleaned up and restarted
        '''
        if self.process is not None:
            retcode = self.process.poll()
            now = datetime.datetime.now()
            minutes = (now - self.start_time).total_seconds()/60
            log.debug('Process pid=%s running with return code %s for %{0.1f} minutes', self.process.pid, retcode, minutes)
            if retcode is not None or minutes > self.values.timeout:
                log.info('Process DIED UNPROVOKED with return code %s or timeout with %0.1f > %0.1f. Cleaning up and restarting', retcode, minutes, self.values.timeout)
                scanid = self.curr_scanid
                self.terminate_process()
                log.info('Process terminated. Restarting scanid %s', scanid)
                self.start_process(scanid)

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
            self.scan_changed(sbid, scanid, target, metadata=None)
            while True:
                time.sleep(1)
                self.poll_process()
                new_scanid = self.scan_pv.get()
                if new_scanid != scanid:
                    self.obsman.scan_changed(sbid, scanid, target, metadata=None)
                    scanid = new_scanid
        except KeyboardInterrupt:
            log.info('Ctrl-C detected')
        except:
            log.exception('Failiure polling process')
        finally:
            self.shutdown()


class MetadataObsmanDriver(MetadataSaver):
    def __init__(self, obsman):
        import Ice
        comm = Ice.initialize(sys.argv)
        super().__init__(comm)
        self.obsman = obsman
        self.sbid = None
        self.scan_running = False

    def changed(self, sbid, state, updated, old_state, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        print(('SB STATE CHANGED', sbid, state, updated, old_state, current))

        if state == ObsState.EXECUTING:
            self.sbid = sbid
            # TODO: pick up antenn  list from pyparset and make antenna mask
            self.scan_manager = ScanManager()
        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None

    def publish(self, pub_data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320

        '''
        if self.sbid is None:
            return

        d = metadata_to_dict(pub_data, self.sbid)
        mgr = self.scan_manager
        next_scan_running = mgr.push_data(d)
        if self.scan_running:
            if next_scan_running: # continue running scan
                pass                
            else: # stop running scan
                self.obsman.terminate_process()
        else:
            if next_scan_running: # start new scan
                self.obsman.scan_changed(self, mgr.sbid, mgr.scan_id, mgr.target_name, self.scan_manager)
            else:
                pass # continue not running a scan
        
        self.scan_running = next_scan_running

    def wait(self):
        self.comm.waitForShutdown()



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
        d = MetadataObsmanDriver(obs)
    else:
        d = EpicsObsmanDriver(obs)        
    d.wait()
    

if __name__ == '__main__':
    _main()
