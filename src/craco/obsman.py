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

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Obsman:
    def __init__(self, values):
        self.scan_pv = PV('ak:md2:scanId_O')
        self.sbid_pv = PV('ak:md2:schedulingblockId_O')
        self.target_pv = PV('ak:md2:targetName_O')
        self.curr_scanid = None
        self.process = None
        
        self.cmd = values.cmd
        self.values = values
        self.doquit = False

        # Bit of a race condition here, but we'll do it
        # add_callback might have beaten us to it,
        #self.scan_changed(self.target_pv.pvname, self.target_pv.get())
        #self.callback_id = self.target_pv.add_callback(self.scan_changed)
        self.scan_changed(self.scan_pv.pvname, self.scan_pv.get())
        self.callback_id = self.scan_pv.add_callback(self.scan_changed)

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
        
    def scan_changed(self, pvname=None, value=None, char_value=None, **kw):
        time.sleep(1) # wait a bit to see if PVs update. #sigh
        new_scanid = self.scan_pv.get()
        sbid = self.sbid_pv.get()
        target = self.target_pv.get() # this doesnt refresh for some reason
        match = None
        if self.values.target_regex is not None:
            match = re.search(self.values.target_regex, target)
            target_ok = match is not None
        else:
            target_ok = True

        log.info(f'Scan_changed pv={pvname} newscanID={new_scanid} currscan={self.curr_scanid} SB{sbid} target={target} OK?={target_ok}')
        if new_scanid == -2 or new_scanid is None: # it's closing - sometimes glitches
            self.terminate_process()
        elif new_scanid == -1: # it's getting ready, do nothign
            pass
        elif new_scanid == self.curr_scanid: # avoid race condition
            pass
        elif target_ok: # new valid scan number with new scan ID
            assert new_scanid >= 0 and new_scanid != self.curr_scanid
            self.start_process(new_scanid)
        else:
            log.info('Passing on %s as doesnt match regex %s', target, self.values.target_regex)

    def start_process(self, new_scanid:int):
        assert new_scanid is not None
        # terminate process to start with
        self.terminate_process()

        assert self.curr_scanid is None
        assert self.process is None
        self.curr_scanid = new_scanid
        
        # start new process group and use it to kill all subprocesses. on exit
        # I love you so much Alexandra
        # https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
        self.process = Popen(self.cmd, shell=True, start_new_session=True)
        pgid = os.getpgid(self.process.pid)
        log.info(f'Started process {self.cmd} with PID={self.process.pid} PGID={pgid} retcode={self.process.returncode}')
        
    def terminate_process(self):
       proc = self.process
       if self.process is not None:
            pgid = os.getpgid(proc.pid) # get process group ID - which we got our own when we started the session
            timeout = self.values.timeout
            log.info(f'sending SIGNINT processes from PID={proc.pid} PGID={pgid}and waiting {timeout} seconds')
            proc.send_signal(signal.SIGINT)
            os.killpg(pgid, signal.SIGINT)
            #proc.terminate()
            try:
                proc.wait(timeout)
            except TimeoutExpired:
                log.warning('Process didnt complete after terminate. Doing kill')
                os.killpg(pgid, signal.SIGKILL)

            proc.wait()
            retcode = proc.returncode
            log.info('Process dead with return code %s. Stopping packets just in case', retcode)
            caput('ak:cracoStop', 1)
            
            # assert retcode is not None
            self.process = None
            self.curr_scanid = None

    def poll_process(self):
        if self.process is not None:
            retcode = self.process.poll()
            log.debug('Process pid=%s running with return code %s', self.process.pid, retcode)
            if retcode is not None:
                log.info('Process terminated with return code %s. Cleaning up', retcode)
                self.terminate_process()

    def shutdown(self):
        log.info('Shutting down process')
        self.scan_pv.remove_callback(self.callback_id)
        self.terminate_process()
        sys.exit()
        
    def wait(self):
        try:
            while True:
                time.sleep(1)
                self.poll_process()
        except KeyboardInterrupt:
            log.info('Ctrl-C detected')
        except:
            log.exception('Failiure polling process')
        finally:
            self.shutdown()


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--timeout', type=int, help='Timeout to wait after sending signal before killing process', default=30)
    parser.add_argument('-R','--target-regex', help='Regex to apply to target name. If match then we start a scan')
    parser.add_argument(dest='cmd', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    obs = Obsman(values)
    obs.wait()
    

if __name__ == '__main__':
    _main()
