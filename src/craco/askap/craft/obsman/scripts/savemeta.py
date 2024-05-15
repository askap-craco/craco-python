#!/usr/bin/env python

import sys
import os
import Ice
from askap import logging
from craco.askap.craft.obsman.metadatasaver import MetadataSaver

def main():
    logger = logging.getLogger(__file__)
    communicator = Ice.initialize(sys.argv)
    #logging.init_logging(sys.argv)
    savedir="/data/TETHYS_1/craftop/metadata_save/"

    try:
        saver = MetadataSaver(communicator,savedir)
        communicator.waitForShutdown()
    except Exception as ex:
        logger.error(str(ex))
        #print(ex, file=sys.stderr)
        #raise
        #sys.exit(1)
    finally:
        saver.close_file()

        

if __name__ == "__main__":
    main()
