#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
from epics import PV
from subprocess import Popen
import time


log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Obsman:
    def __init__(self, values):
        self.scan_pv = PV('ak:md2:scanId_O')
        self.sbid_pv = PV('ak:md2:schedulingblockId_O')
        self.callback_id = self.scan_pv.add_callback(self.scan_changed)
        self.curr_scanid = None
        self.process = None
        
        self.cmd = values.cmd
        self.values = values

        # Bit of a race condition here, but we'll do it
        # add_callback might have beaten us to it,
        self.scan_changed(self.scan_pv.pvname, self.scan_pv.get())


    def scan_changed(self, pvname=None, value=None, char_value=None, **kw):
        assert pvname == self.scan_pv.pvname
        new_scanid = value
        sbid = self.sbid_pv.get()
        log.info(f'Scan_changed pv={pvname} newscan={value} currscan={self.curr_scanid} SB{sbid}')
        if new_scanid == -2: # it's closing
            self.terminate_process()
        elif new_scanid == -1: # it's getting ready, do nothign
            pass
        elif new_scanid == self.curr_scanid: # avoid race condition
            pass
        else: # new valid scan number with new scan ID
            assert new_scanid >= 0 and new_scanid != self.curr_scanid
            self.start_process(new_scanid)


    def start_process(self, new_scanid:int):
        # terminate process to start with
        self.terminate_process()

        assert self.curr_scanid is None
        assert self.process is None
        self.curr_scanid = new_scanid
        self.process = Popen(self.cmd, shell=True)
        log.info(f'Started process {self.cmd} with PID={self.process.pid} retcode={self.process.returncode}')
        
    def terminate_process(self):
        proc = self.process
        if self.process is not None:
            log.info('Killing process')
            proc.terminate()
            proc.wait()
            retcode = proc.returncode
            log.info('Process dead with return code %s', retcode)
            # assert retcode is not None
            self.process = None
            self.curr_scanid = None

    def poll_process(self):
        if self.process is not None:
            log.debug('Process pid=%s running with return code %s', self.process.pid, self.process.returncode)

    def shutdown(self):
        self.scan_pv.remove_callback(self.callback_id)
        self.terminate_process()
        
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
