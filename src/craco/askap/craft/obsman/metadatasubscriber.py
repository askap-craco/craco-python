#!/usr/bin/env python
#
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
import sys
import os
import Ice
import IceStorm

# pylint: disable-msg=W0611
from askap.slice import TypedValues

# ice doesn't agree with pylint
# pylint: disable-msg=E0611
import askap.interfaces as iceint
from askap.interfaces.datapublisher import (ITimeTaggedTypedValueMapPublisher,
                                            ITimeTaggedTypedValueMapPublisherPrx)
from askap.interfaces.datapublisher import (ITypedValueMapPublisher,
                                            ITypedValueMapPublisherPrx)


# noinspection PyUnusedLocal,PyMethodMayBeStatic
class MetadataImpl(ITimeTaggedTypedValueMapPublisher):
    def publish(self, data, current=None):
        print(("-"*80))
        print((data.timestamp, type(data.timestamp)))
        for k in sorted(data.data.keys()):
            print((k, data.data[k].value))


class MetadataSubscriber(object):
    """IceStorm publisher for TOS metadata

    :topics:
    metadata1 == beamformer = 1 second
    metadtat2 == correaltor = 5-10 seconds.
    """
    def __init__(self, ice, metadata_impl, topic='metadata2'):
        if ice is None:
            self.ice = self._setup_communicator()
        else:
            self.ice = ice

        if metadata_impl is None:
            self.metadata_impl = MetadataImpl()
        else:
            self.metadata_impl = metadata_impl

        self.prxy = None
        self.manager = None
        self._topic = topic
        self._setup_icestorm()

    @staticmethod
    def _setup_communicator():
        if "ICE_CONFIG" in os.environ:
            return Ice.initialize(sys.argv)
        host = 'icehost-mro.atnf.csiro.au'
        port = 4061
        init = Ice.InitializationData()
        init.properties = Ice.createProperties()
        loc = "IceGrid/Locator:tcp -h "+ host + " -p " + str(port)
        init.properties.setProperty('Ice.Default.Locator', loc)
        init.properties.setProperty('Ice.IPv6', '0')
        return Ice.initialize(init)

    def _setup_icestorm(self):
        """Create the IceStorm connection and subscribe to the logger topic.
        """
        if not self.manager:
            prxstr = self.ice.stringToProxy(
                'IceStorm/TopicManager@IceStorm.TopicManager')
            try:
                self.manager = IceStorm.TopicManagerPrx.checkedCast(prxstr)
            except (Ice.LocalException, Exception) as ex:
                self.manager = None
                raise ex
        try:
            self.topic = self.manager.retrieve(self._topic)
        except IceStorm.NoSuchTopic:
            try:
                self.topic = self.manager.create(self._topic)
            except IceStorm.TopicExists:
                self.topic = self.manager.retrieve(topicname)

        self.adapter = \
            self.ice.createObjectAdapterWithEndpoints("MetadataAdapter",
                                                      "tcp")
        self.subscriber = self.adapter.addWithUUID(self.metadata_impl).ice_oneway()
        qos = {}
        try:
            self.topic.subscribeAndGetPublisher(qos, self.subscriber)
        except IceStorm.AlreadySubscribed:
            self.topic.unsubscribe(self.subscriber)
            self.topic.subscribeAndGetPublisher(qos, self.subscriber)
        self.adapter.activate()


if __name__ == "__main__":
    topic_name = "metadata"
    if len(sys.argv) > 1:
        topic_name = sys.argv[1]
    msub = MetadataSubscriber(topic_name)
    msub.ice.waitForShutdown()
    msub.topic.unsubscribe(self.subscriber)
