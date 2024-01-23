# Copyright (c) 2016 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
#

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

# pylint: disable-msg=W0611
from .metadatasubscriber import MetadataSubscriber
from .sbstatemonitor import SBStateSubscriber
from askap.parset import parset_to_dict
from askap import logging

import askap.interfaces as iceint
from askap.interfaces.schedblock import ObsState
from askap.interfaces import Direction, TypedValueType
from askap.iceutils import get_service_object
import json
from epics import PV
import os
import gzip


logger = logging.getLogger(__name__)

def coerceice(v):
    if isinstance(v.value, Direction):
        vout = (v.value.coord1, v.value.coord2, str(v.value.sys))
    elif v.type == TypedValueType.TypeDirectionSeq:
        vout = [(c.coord1, c.coord2, str(c.sys)) for c in v.value]
    else:
        vout = v.value

    return vout

class MetadataSaver(iceint.schedblock.ISBStateMonitor,
                   iceint.datapublisher.ITimeTaggedTypedValueMapPublisher):
    """The Craft Manager application class.

    """
    def __init__(self, comm, savedir="."):
        self._sb = get_service_object(comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)

        sbmgr = self._sb
        executing_block_ids = sbmgr.getByState([ObsState.EXECUTING], None)
        print(('Got', len(executing_block_ids), 'Executing blocks', executing_block_ids))
        self.fout = None
        self.savedir = savedir
        if len(executing_block_ids) == 1:
            self.changed(executing_block_ids[0], ObsState.EXECUTING, None,None)
        else:
            self.sbid = None
            self.state = None


        self.sb_sub = SBStateSubscriber(comm, self)
        self.metadata_sub = MetadataSubscriber(comm, self)
        self.ant_state_pvs = {}


    def open_file(self, sbid):
        if self.fout is not None:
            self.close_file()

        fname = os.path.join(self.savedir, f'SB{sbid:d}.json.gz')
        self.fout = gzip.open(fname, 'at')
        print('Opened file', self.fout)
        return self.fout

    def close_file(self):
        if self.fout is not None:
            self.fout.close()
            print('Closed file', self.fout)
            self.fout = None

    def __del__(self):
        self.close_file()

    def changed(self, sbid, state, updated, old_state, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        print(('SB STATE CHANGED', sbid, state, updated, old_state, current))

        if state == ObsState.EXECUTING:
            self.sbid = sbid
            self.open_file(sbid)
        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None
            self.close_file()


    def publish(self, pub_data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320

        '''
        if self.sbid is None:
            return
        
        ts = pub_data.timestamp
        data = pub_data.data
        d = {}
        d['timestamp'] = ts
        d['sbid'] = self.sbid
        ant_data = {}
        d['antennas'] = ant_data
        # Make new dictionary of vanilla python types and make it a hierarchy so it'll play nicer with JSON
        for k,v in list(data.items()):
            if k == 'antennas':
                antennas = coerceice(v)
            elif k.startswith('ak') or k.startswith('co'):
                ksplit = k.split('.')
                if len(ksplit) != 2:
                    continue
                    
                antname, data_key = k.split('.')
                if antname not in list(ant_data.keys()):
                    ant_data[antname] = {}

                ant_data[antname][data_key] = coerceice(v)
            else:
                d[k] = coerceice(v)

        jsons = json.dumps(d, sort_keys=True, indent=4)
        if self.fout is not None:
            try:
                json.dump(d, self.fout, sort_keys=True, indent=4)
                self.fout.flush()
            except Exception as e:
                logger.error('Error writing data')
                print('Error writing data', e)
                self.close_file()
            
        #jout = json.dumps(d)

                
                

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-D','--destdir', required=True, help='Destination directory')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    import Ice
    import sys
    communicator = Ice.initialize(sys.argv)
    savedir=values.destdir
    saver = None

    try:
        saver = MetadataSaver(communicator,savedir)
        communicator.waitForShutdown()
    except Exception as ex:
        logger.exception('Error saving data')
    finally:
        if saver is not None:
            saver.close_file()


        


        
        
