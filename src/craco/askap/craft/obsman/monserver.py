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

'''
Server thaft gets metadata data from monica
'''

import time
from askap.slice import CraftService, CommonTypes
from askap.interfaces import Direction
from askap.interfaces import CoordSys
import askap.interfaces as iceint
from askap.iceutils import Server
from . import logger
from .impl import CraftManagerImpl
from epics import PV

def mon2direction(monvalue):
    ''' Turns a monical value into a direction'''
    bits = list(map(float, monvalue.value.split()))
    d = Direction(*bits)
    return d

def floatnone(s):
    try:
        return float(s)
    except:
        return None

def intnone(s):
    try:
        t = iceint.TypedValueInt()
        t.type = iceint.TypedValueType.TypeInt
        t.value = int(s)
        return t
    except:
        return None

def icestr(s):
    t = iceint.TypedValueString()
    t.type = iceint.TypedValueType.TypeString
    t.value = s
    
    return t

def icestrlist(slist):
    t = iceint.TypedValueStringSeq()
    t.type = iceint.TypedValueType.TypeStringSeq
    t.value = [list(map(icestr, slist))]
    
    return t
    


class MonicaCraftManager(Server):
    def __init__(self, comm, mon, ants, values):
        Server.__init__(self, comm, fcmkey='askap.craft')
        self.logger = logger
        self.monitoring = False
        self._comm = comm

        self._mon = mon
        self._points = []
        self._addpoints('site.schedblock','id','alias', 'scan','duration','progress', 'template')
        self._addpoints('site.schedblock.target', 'direction', 'frequency', 'phase_direction', 'name')
        self._addpoints('site.schedblock.target.footprint', 'name','pitch','rotation')

        for ant in ants:
            self._addpoints('ak%02d.servo' % ant, 'coords.RA','coords.Dec','coords.Pol', 'State','SubState', 'coords.Az', 'coords.El')
            
        self._ants = ants
        self._server = None
        self._curr_sbid = None
        self._server = CraftManagerImpl(self._comm, self.parameters, values)
        self._test = values.test
        self._test_timestamp = None
        self._par_angle_pv = {}

    def initialize_services(self):
        """Base class override, which adds the service implementation
        instance"""

        self.add_service("CraftManager", self._server)

    def _addpoints(self, root, *subs):
        if subs is None or len(subs) == 0:
            self._addpoint(root)
        else:
            for sub in subs:
                self._addpoint(root+ '.' + sub)

    def _addpoint(self, point):
        self._points.append(point)
        # get details - client cches automatgivally
        self._mon.details(point)


    def _poll(self):
        mondata = self._mon.poll(self._points)
        metadata = self._make_metadata(mondata)
        return metadata

    def _make_test_metadata(self):
        if self._test_timestamp is None:
            self._test_timestamp = 0
        else:
            self._test_timestamp += 1

        d = {}
        d['_schedblock_id'] = 2981
        d['scan_id'] = 0
        d['sky_frequency'] = 1278.5
        d['antennas'] = icestrlist(['ak%02d' % ant for ant in self._ants])
        d['flagged'] = False
        #d['target_direction'] = mon2direction(mondata['site.schedblock.target.direction'])
        #d['phase_direction'] = mon2direction(mondata['site.schedblock.phase_direction'])

        d['corrmode'] = 'standard'
        
        # TODO - a hopless hack to get things going - monica doesn't seem to have the relevant
        # target name
        d['target_name'] = icestr('src1' )
        
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


    def get_par_angle(self, antid):
        if antid not in list(self._par_angle_pv.keys()):
            pv = PV('ak{:02d}:drives:skyPol'.format(antid))
            self._par_angle_pv[antid] = pv
            
        return float(self._par_angle_pv[antid].get())

    def _make_metadata(self, mondata):
        '''
        # Assemble data into the correct format
        #:data: is a directionary whose contents is defined here:
        # https://jira.csiro.au/browse/ASKAPTOS-3320 

        :returns: a dict whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320
        '''
        d = {}
        d['_schedblock_id'] = intnone(mondata['site.schedblock.id'].value)

        print(d)

        
        d['_schedblock_alias'] = mondata['site.schedblock.alias'].value
        d['_schedblock_template'] = mondata['site.schedblock.template'].value
        d['scan_id'] = intnone(mondata['site.schedblock.scan'].value)
        d['sky_frequency'] = floatnone(mondata['site.schedblock.target.frequency'].value)
        d['antennas'] = icestrlist(['ak%02d' % ant for ant in self._ants])
        d['flagged'] = False
        #d['target_direction'] = mon2direction(mondata['site.schedblock.target.direction'])
        #d['phase_direction'] = mon2direction(mondata['site.schedblock.phase_direction'])

        d['corrmode'] = 'standard'
        
        # TODO - a hopless hack to get things going - monica doesn't seem to have the relevant
        # target name
        d['target_name'] = icestr('src1' )
        d['_target_srcname'] = mondata['site.schedblock.target.name'].value
        d['_footprint_name'] = mondata['site.schedblock.target.footprint.name'].value
        d['_footprint_pitch'] = mondata['site.schedblock.target.footprint.pitch'].value
        d['_footprint_rotation'] = mondata['site.schedblock.target.footprint.rotation'].value

        timestamp = 0
        

        for ant in self._ants:
            ap = 'ak%02d.' % ant
            tracking = mondata[ap+'servo.SubState'].value == 'Tracking'
            d[ap+'flagged'] = not tracking
            d[ap+'on_source'] = tracking
            d[ap+'actual_pol'] = floatnone(mondata[ap+'servo.coords.Pol'].value)
            d[ap+'par_angle'] = self.get_par_angle(ant)
            az = floatnone(mondata[ap+'servo.coords.Az'].value)
            el = floatnone(mondata[ap+'servo.coords.El'].value)
            ra = floatnone(mondata[ap+'servo.coords.RA'].value)
            dec = floatnone(mondata[ap+'servo.coords.Dec'].value)
        
            d[ap+'actual_azel'] = Direction(az, el, CoordSys.AZEL)
            d[ap+'actual_radec'] = Direction(ra, dec, CoordSys.J2000)
            rabat = mondata[ap+'servo.coords.RA'].bat
            if rabat > timestamp:
                rabat = timestamp

        return timestamp, d

    def run_single(self):
        if self._test:
            timestamp, curr_data = self._make_test_metadata()
        else:
            timestamp, curr_data = self._poll()

        sbid = curr_data['_schedblock_id']
        if sbid is None:
           if self._curr_sbid is None: # Nothing changed - nothign happening
               pass
           else: # No SBID any more. WE're finisehd
               self._server.abortObs()
        else: 
            if sbid == self._curr_sbid: # nothign changed - current sbid
                pass
            elif self._curr_sbid is None: # first poll with a new sbid
                self._server.startObs(sbid.value)
            else: # New pol has a different sbid to the previous one
                assert(sbid != self._curr_sbid)
                self._server.abortObs()
                self._server.startObs(sbid.value)

        self._curr_sbid = sbid
        self._server.on_metadata(timestamp, curr_data)

    def run(self):
        while True:
            self.run_single()
            time.sleep(5)
