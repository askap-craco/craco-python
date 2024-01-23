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
# for slack
import requests
import json
import socket

from askap.iceutils import get_service_object
from askap.slice import SchedulingBlockService

# pylint: disable-msg=E0611
import askap.interfaces as iceint

class SBStateMonitorImpl(iceint.schedblock.ISBStateMonitor):

    def __init__(self, errorfunc):
        self.errorfunc = errorfunc

    # pylint: disable-msg=W0613,W0603,R0201
    def changed(self, sbid, state, updated, old_state, current=None):
        print(("Status change:", sbid, state, 'from', old_state))
        message = {"username": "SB Status",
                   "text": "*SB{}* transitioned to {} from {}".format(
                       sbid, state, old_state)}
        if str(state).startswith("ERROR"):
            msg = self.errorfunc(sbid)
            attachment = {
                "fallback": msg,
                "color": "danger",
                "text": "```{}```".format(msg),
                "mrkdwn_in": ["text"]
            }
            message["attachments"] = [attachment]

        jmessage = json.dumps(message)
        r = requests.post(SLACK_URL[HOST], jmessage,
                          headers={'content-type': 'application/json'})
        print((r.status_code, r.text))


class SBStateSubscriber(object):
    def __init__(self, comm, monitor_impl):
        self.topic_name = "sbstatechange"
        self.manager = None
        if comm is None:
            self.ice = self._setup_communicator()
        else:
            self.ice = comm

        if monitor_impl is None:
            self.monitor_impl = SBStateMonitorImpl(self.get_error)
        else:
            self.monitor_impl = monitor_impl

        self._setup_subscriber()


    def _setup_subscriber(self):
        self.manager = get_service_object(
            self.ice,
            'IceStorm/TopicManager@IceStorm.TopicManager',
            IceStorm.TopicManagerPrx
        )
        try:
            self.topic = self.manager.retrieve(self.topic_name)
        except IceStorm.NoSuchTopic:
            try:
                self.topic = self.manager.create(self.topic_name)
            except IceStorm.TopicExists:
                self.topic = self.manager.retrieve(self.topic_name)
        # defined in config.icegrid
        self.adapter = \
            self.ice.createObjectAdapterWithEndpoints("SBStateSubAdapter",
                                                      "tcp")

        self.subscriber = self.adapter.addWithUUID(self.monitor_impl).ice_oneway()
        qos = {}
        try:
            self.topic.subscribeAndGetPublisher(qos, self.subscriber)
        except IceStorm.AlreadySubscribed:
            raise
        self.adapter.activate()


    @staticmethod
    def _setup_communicator():
        if "ICE_CONFIG" in os.environ:
            return Ice.initialize(sys.argv)
        host = 'localhost'
        port = 4062
        init = Ice.InitializationData()
        init.properties = Ice.createProperties()
        loc = "IceGrid/Locator:tcp -h " + host + " -p " + str(port)
        init.properties.setProperty('Ice.Default.Locator', loc)
        init.properties.setProperty('Ice.IPv6', '0')
        return Ice.initialize(init)


if __name__ == "__main__":
    state = SBStateSubscriber()
    try:        
        state.ice.waitForShutdown()
    except KeyboardInterrupt:
        state.topic.unsubscribe(state.subscriber)
        state.ice.shutdown()
