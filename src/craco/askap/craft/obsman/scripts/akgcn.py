#!/usr/bin/env python

import sys
import os
import Ice
from askap import logging
from askap.craft.obsman.server import CraftManager

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='ASKAP GCN listerner')
    parser.add_argument('-t','--test', action='store_true', help='Make test data, not real data', default=False)
    parser.add_argument('--log-config', help='log config')
    values = parser.parse_args()


    logger = logging.getLogger(__file__)
    communicator = Ice.initialize(sys.argv)
    logging.init_logging(sys.argv)
    print(os.environ['ICE_CONFIG'])
    try:
        gcc = GcnListener(communicator, values)
        communicator.waitForShutdown()
    except Exception as ex:
        logger.error(str(ex))
        print(ex, file=sys.stderr)
        raise
        sys.exit(1)
        

if __name__ == "__main__":
    main()
