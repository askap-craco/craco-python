#!/usr/bin/env python
#
# @copyright (c) 2016 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
# @author Craig.Haskins@csiro.au
#
'''
ADE Beamformer - setup CRAFT download
'''
from askap.osl.scripting import AutoScript, ui
from askap.osl.subsystems import Ade
import numpy as np

class SetupCraft(Ade):
    '''
    ADE Beamformer - setup CRAFT download
    '''
    def __init__(self, **kw):
        Ade.__init__(self, **kw)
        self.figures = []

    def run(self):
        pset = self.parSet.ade_bmf
        bmflist = self.dbe.bmfList
        bats = [bm.get_current_bat() for bm in bmflist]
        maxbat = max(bats)
        minbat = min(bats)
        print(('Got', len(bats), 'bats, of which the max is', maxbat, minbat, [bat - minbat for bat in bats]))
        int_time = int(pset.craft.int_time)
        inttime_usec = int(27./32. * float(int_time+1))
        int_count = int(pset.craft.int_count)
        packet_interval_usecs = float((int_time + 1)*int_count)*32./27.
        interval_per_ant_bat = int(packet_interval_usecs / float(len(bats))) * 0
        nant = len(bats)

        # assign antennas to integrations sequentially
        setup_bats = [maxbat + pset.craft.event_delay*1000000 + 0*(iant % int_count)*inttime_usec for iant in range(nant)]
        print(('INT_COUNT {} INT_TIME{} ={} usec packet interval usecs {} interval_per_ant_bat {}'.format(int_count, int_time, inttime_usec, packet_interval_usecs, interval_per_ant_bat)))

        for ibmf, (bmf, this_setup_bat) in enumerate(zip(bmflist, setup_bats)):
            actual_start_bat = bmf.setupCraft(int_time, int_count, pset.craft.mode, pset.craft.beam, False, this_setup_bat, wait=False)
            print(('Setup bmf {} with bat offset {} actual bat offset {}'.format(ibmf, this_setup_bat - maxbat, actual_start_bat - maxbat)))

        bats = [bm.get_current_bat() for bm in bmflist]
        print(('Got', len(bats), 'bats, of which the max is', maxbat, minbat, [bat - minbat for bat in bats]))
        for ibat, bat in enumerate(bats):
            if bat >  max(setup_bats):
                print(('AAAARGH!!! Running late! Increase event_delay and try again', ibat, bat, max(setup_bats)))

        return True

if __name__ == '__main__':
    myScript = SetupCraft()
    myScript.start()
