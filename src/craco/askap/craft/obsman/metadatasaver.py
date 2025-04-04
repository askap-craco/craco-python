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
from .metadatasubscriber import MetadataSubscriber, metadata_to_dict
from .sbstatemonitor import SBStateSubscriber
from askap.parset import parset_to_dict
from askap import logging
import askap.slice
import askap.iceutils # needs to be imported before interfaces

import askap.interfaces as iceint
from askap.interfaces.schedblock import ObsState
from askap.interfaces import Direction, TypedValueType
from askap.iceutils import get_service_object


import json
from epics import PV
import os
import gzip
import datetime
from astropy.time import Time


logger = logging.getLogger(__name__)

class FileWriterMetadataListener:
    def __init__(self, savedir="."):
        self.fout = None
        self.savedir = savedir

    def open_file(self, sbid):
        if self.fout is not None:
            self.close_file()

        fname = os.path.join(self.savedir, f'SB{sbid:d}.json.gz')
        self.fout = gzip.open(fname, 'at')
        logger.info('Opened file %s', self.fout)
        return self.fout

    def close_file(self):
        if self.fout is not None:
            self.fout.close()
            logger.info('Closed file %s', self.fout)
            self.fout = None

    def changed(self, sbid, state, updated, old_state, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''

        if state == ObsState.EXECUTING:
            self.sbid = sbid
            self.open_file(sbid)
        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None
            self.close_file()

    def publish(self, pub_data, current=None):
        '''
        Recieve metdadta.
        d is a dictionary version of the metadata which is easier to digest in json
        it also has the timestamp and sbid in it
        '''            

        if self.fout is not None:
            try:
                d = metadata_to_dict(pub_data, self.sbid)

                jsons = json.dumps(d, sort_keys=True, indent=4)
                json.dump(d, self.fout, sort_keys=True, indent=4)
                self.fout.flush()
            except Exception as e:
                logger.error('Error writing data')
                print('Error writing data', e)
                self.close_file()
            

    close = close_file

    def __del__(self):
        self.close_file()


class MetadataSaver(iceint.schedblock.ISBStateMonitor,
                   iceint.datapublisher.ITimeTaggedTypedValueMapPublisher):
    """The Craft Manager application class.

    """
    def __init__(self, comm, listener=None):
        self._sb = get_service_object(comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)

        sbmgr = self._sb
        executing_block_ids = sbmgr.getByState([ObsState.EXECUTING], None)
        logger.debug('Got %d executing blocks: %s', len(executing_block_ids), executing_block_ids)
        self.listener = listener

        if len(executing_block_ids) == 1:
            self.changed(executing_block_ids[0], ObsState.EXECUTING, None,None)
        else:
            self.sbid = None
            self.state = None

        self.sb_sub = SBStateSubscriber(comm, self)
        self.metadata_sub = MetadataSubscriber(comm, self)
        self.sbid = None
        if listener is not None:
            listener.sb_service = self._sb

    def changed(self, sbid, state, updated, old_state, current=None):
        '''Implements ISBStateMonitor
        Called when schedblock state changes
        '''
        logger.debug('SB STATE CHANGED sbid=%s state=%s updated=%s old_state=%s', sbid, state, updated, old_state)

        # keep track of SBID
        if state == ObsState.EXECUTING:
            self.sbid = sbid
        elif sbid == self.sbid:
            assert state != ObsState.EXECUTING
            # It must have gone out of executing
            self.sbid = None

        if self.listener is not None:
            self.listener.changed(sbid, state, updated, old_state, current)

    def publish(self, pub_data, current=None):
        '''Implements iceint.datapublisher.ITimeTaggedTypedValueMapPublisher
        Called when new metadata received
        :data: is a directionary whose contents is defined here: https://jira.csiro.au/browse/ASKAPTOS-3320

        '''

        ts = Time(pub_data.timestamp/1e6/3600/24, format='mjd', scale='tai')
        now = Time.now()
        
        logger.debug('Got metadata now=%s ts=%s difference %0.1f seconds',now.iso, ts.iso, (now - ts).to('second').value )

        if self.listener is not None:
            self.listener.publish(pub_data, current)

    def close(self):
        if self.listener is not None:
            self.listener.close()
                
                
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-D','--destdir', required=True, help='Destination directory')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    FORMAT = '%(levelname)s %(asctime)s %(module)s %(message)s'
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT)

    import Ice
    import sys
    communicator = Ice.initialize(sys.argv)
    savedir=values.destdir
    saver = None

    try:
        listener = FileWriterMetadataListener(savedir)
        saver = MetadataSaver(communicator,listener=listener)
        communicator.waitForShutdown()
    except Exception as ex:
        logger.exception('Error saving data')
    finally:
        if saver is not None:
            saver.close()


        


        
        
