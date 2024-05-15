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
import socket


logger = logging.getLogger(__name__)

def coerceice(v):
    if isinstance(v.value, Direction):
        vout = (v.value.coord1, v.value.coord2, str(v.value.sys))
    elif v.type == TypedValueType.TypeDirectionSeq:
        vout = [(c.coord1, c.coord2, str(c.sys)) for c in v.value]
    else:
        vout = v.value

    return vout

class MetadataPusher(iceint.schedblock.ISBStateMonitor,
                   iceint.datapublisher.ITimeTaggedTypedValueMapPublisher):
    """The Craft Manager application class.

    """
    def __init__(self, comm, hostport=None):
        self._sb = get_service_object(comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)

        sbmgr = self._sb
        executing_block_ids = sbmgr.getByState([ObsState.EXECUTING], None)
        print(('Got', len(executing_block_ids), 'Executing blocks', executing_block_ids))
        if len(executing_block_ids) == 1:
            self.changed(executing_block_ids[0], ObsState.EXECUTING, None)
        else:
            self.sbid = None
            self.state = None

        self.sb_sub = SBStateSubscriber(comm, self)
        self.metadata_sub = MetadataSubscriber(comm, self)
        self.ant_state_pvs = {}
        self.hostport = hostport
        if self.hostport is not None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            self.sock = None
            

    def changed(self, sbid, state, updated, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        print(('SB STATE CHANGED', sbid, state, updated, current))

        if state == ObsState.EXECUTING:
            self.sbid = sbid
        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None

    def get_ant_state(self, antname):
        if antname not in list(self.ant_state_pvs.keys()):
            self.ant_state_pvs[antname] = PV(antname+':drives:subState')

        state = self.ant_state_pvs[antname].get()
        return state


    def publish(self, pub_data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320

        '''
        if self.sbid is None:
            return
        
        ts = pub_data.timestamp
        data = pub_data.data
        print(data)
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

        #print json.dumps(d, sort_keys=True, indent=4)
        #jout = json.dumps(d)
        tout = self.format_as_text(d)
        logger.debug('Text dump is %d bytes= %s', len(tout), tout)
        if self.sock is not None:
            self.sock.sendto(tout, self.hostport)

    def format_as_text(self, d):
        cards1 = ['{} {}'.format(h, d[h]) for h in ('sbid','flagged','scan_id')]
        antcards  = ['{} {} {} {} {} {}'.format(antname, antd['actual_radec'][0], antd['actual_radec'][1], antd['actual_radec'][2], antd['on_source'], self.get_ant_state(antname)) for antname, antd in list(d['antennas'].items()) if antd['flagged'] == False]

        all_cards = cards1[:]
        all_cards.extend(antcards)
        s = '\n'.join(all_cards)
        s += '\n'
        return s
                
                

    


        
        
