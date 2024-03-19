#!/usr/bin/env python

import sys
import os
import Ice
from askap import logging
from askap.craft.obsman.server import CraftManager
from askap.craft.cmdline import strrange

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Craft observation managed')
    parser.add_argument('-t','--test', action='store_true', help='Make test data, not real data', default=False)
    parser.add_argument('-a','--search-antennas', type=strrange, help='Only start search using these antennas')
    #parser.add_argument('--trigger-all-antennas', action='store_true', default=False, help='If spefified -a, download voltages from all antennas')
    parser.add_argument('--log-config', help='log config')
    parser.add_argument('-p','--extra-parset', help='Overrride the Sb parset with parameters from this parset')
    values = parser.parse_args()

    logger = logging.getLogger(__file__)
    communicator = Ice.initialize(sys.argv)
    logging.init_logging(sys.argv)
    print((os.environ['ICE_CONFIG']))
    craft = CraftManager(communicator, values)
    try:
        communicator.waitForShutdown()
    except KeyboardInterrupt as ex:
        logging.info('Keyboard interrupt. Aborting CRAFT')
        craft.abort()
    except Exception as ex:
        logger.info('Aborting CRAFT from craft.py exception handler')
        craft.abort()
        logger.error(str(ex))
        raise
        sys.exit(1)
        

if __name__ == "__main__":
    main()
