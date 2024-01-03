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
from .impl import CraftManagerImpl
from .metadatasubscriber import MetadataSubscriber
from .sbstatemonitor import SBStateSubscriber
from askap.parset import parset_to_dict
from askap import logging
import threading

import askap.interfaces as iceint
from askap.interfaces.schedblock import ObsState


logger = logging.getLogger(__name__)

class CraftManager(iceint.schedblock.ISBStateMonitor,
                   iceint.datapublisher.ITimeTaggedTypedValueMapPublisher):
    """The Craft Manager application class.

    """
    def __init__(self, comm, values):
        self._lock = threading.Lock()
        with(self._lock):
            self.sb_sub = SBStateSubscriber(comm, self)
            self.metadata_sub = MetadataSubscriber(comm, self)
            self.impl = CraftManagerImpl(comm, values)
            self.curr_sbid = None
            self.values = values
            sbmgr = self.impl._sb
            executing_block_ids = sbmgr.getByState([ObsState.EXECUTING], None)
            print(('Got', len(executing_block_ids), 'Executing blocks', executing_block_ids))
            self.test_timestamp = None
            
        if len(executing_block_ids) == 1:
            self.changed(executing_block_ids[0], ObsState.EXECUTING, None)

    def _make_test_metadata(self):
        if self._test_timestamp is None:
            self._test_timestamp = 0
        else:
            self._test_timestamp += 1

        d = {}
        d['_schedblock_id'] = 999
        d['scan_id'] = 0
        d['sky_frequency'] = 1320.5
        d['antennas'] = ['ak%02d' % ant for ant in self._ants]
        d['flagged'] = False
        d['test'] = True
        d['corrmode'] = 'standard'
        
        # TODO - a hopless hack to get things going - monica doesn't seem to have the relevant
        # target name
        d['target_name'] = 'src1' 
        timestamp = self._test_timestamp

        for ant in self._ants:
            ap = 'ak%02d.' % ant
            tracking = True
            d[ap+'flagged'] = not tracking
            d[ap+'on_source'] = tracking
            d[ap+'actual_pol'] = 0.
            d[ap+'par_angle'] = 0.
            az = 0.
            el = 0.
            ra = 0.
            dec = 0.
        
            d[ap+'actual_azel'] = Direction(az, el, CoordSys.AZEL)
            d[ap+'actual_radec'] = Direction(ra, dec, CoordSys.J2000)
            rabat = 0.
            if rabat > timestamp:
                rabat = timestamp
        

        return timestamp, d

    def abort(self):
        with self._lock:
            self.impl.abortObs()

    def changed(self, sbid, state, updated, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        print(('SB STATE CHANGED', sbid, state, updated, current, self.curr_sbid))
        with self._lock:
            if state == ObsState.EXECUTING:
                if self.curr_sbid is not None:
                    self.impl.abortObs()

                self.curr_sbid = sbid
                self.impl.startObs(sbid)
            elif sbid == self.curr_sbid:
                assert state != ObsState.EXECUTING
                # It must have gone out of executing
                self.impl.abortObs()
                self.curr_sbid = None

    def publish(self, data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        '''
        with self._lock:
            if self.curr_sbid:
                self.impl.on_metadata(data.timestamp, data.data)


        
        
