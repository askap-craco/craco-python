#!/usr/bin/env python
"""
Runs a command when the schedblock status changes

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import subprocess

import Ice
from askap.iceutils import get_service_object
from askap.slice import SchedulingBlockService
from craco.askap.craft.obsman.sbstatemonitor import SBStateSubscriber
from askap.parset import ParameterSet


# pylint: disable-msg=E0611
import askap.interfaces as iceint

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class SBRunner(iceint.schedblock.ISBStateMonitor):
    def __init__(self, errorfunc, cmd):
        self.errorfunc = errorfunc
        self.cmd = ' '.join(cmd)

        env = {'SB_ID':0, 'SB_STATE': 'OBSERVED', 'SB_UPDATED':None, 'SB_OLD_STATE':'EXECUTING'}
        cmd = self.cmd.format(**env)
        log.info('Will be running cmd env that looks like %s %s',env, cmd)

    def changed(self, sbid, state, updated, old_state, current=None):
        log.info(("Status change:", sbid, state, 'from', old_state))
        env = {}
        env['SB_ID'] = sbid
        env['SB_STATE'] = state
        env['SB_UPDATED'] = updated
        env['SB_OLD_STATE'] = old_state

        if old_state == 'EXECUTING':
            try:
                cmd = self.cmd.format(**env)
                log.info(f'Calling check called with {cmd} and env {env}')
                env.update(os.environ)
                subprocess.check_call(cmd, env=env)
                log.info('CMD %s was successfull', cmd)
            except subprocess.CalledProcessError as e:
                log.error('CMD %s failed with error code %d', cmd, e.returncode)
            
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    h = """
    "Runs the command from the commandline at the end of the schedblock.
    Currenly only runs if the old_state was 'EXECUTING'.
    Adds SB_ID, SB_STATE, SB_UPDATED and SB_OLD_STATE to the evnrionment, and also interprets the command line if it contained {SB_ID} keys etc.

    example: sbrunner ./prepare_skadi.py -cal $cal -obs {SB_ID}
    """
    parser = ArgumentParser(description=h, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='cmd', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # mannaully get a communicator here
    host = 'icehost-mro.atnf.csiro.au'
    port = 4061
    timeout_ms = 5000
    default_loc = "IceGrid/Locator:tcp -h " + host + " -p " + str(port) + " -t " + str(timeout_ms)

    init = Ice.InitializationData()
    init.properties = Ice.createProperties()
    if "ICE_CONFIG" not in os.environ:
        loc = default_loc
    else:
        ice_cfg_file = os.environ['ICE_CONFIG']
        ice_parset = ParameterSet(ice_cfg_file)
        loc = ice_parset.get_value('Ice.Default.Locator', default_loc)

    init.properties.setProperty('Ice.Default.Locator', loc)
    _communicator = Ice.initialize(init)
    
    cmd = values.cmd
    runner = SBRunner(None, cmd)
    state = SBStateSubscriber(_communicator, runner)
    try:        
        state.ice.waitForShutdown()
    except KeyboardInterrupt:
        state.topic.unsubscribe(state.subscriber)
        state.ice.shutdown()

    

if __name__ == '__main__':
    _main()
