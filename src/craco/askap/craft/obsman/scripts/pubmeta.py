#!/usr/bin/env python

import sys
import os
import Ice
from askap import logging
from askap.craft.obsman.metadata2udpserver import MetadataPusher

def main():
    logger = logging.getLogger(__file__)
    communicator = Ice.initialize(sys.argv)
    #logging.init_logging(sys.argv)
    hostport = ('localhost', 1234)
    hostport = ('telemetry.mwa128t.org', 54321)
    try:
        craft = MetadataPusher(communicator, hostport)
        communicator.waitForShutdown()
    except Exception as ex:
        logger.error(str(ex))
        print(ex, file=sys.stderr)
        raise
        sys.exit(1)
        

if __name__ == "__main__":
    main()
