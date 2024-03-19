#!/usr/bin/env python

from askap import logging
from askap.craft.obsman.monserver import MonicaCraftManager
from askap.craft.monica import MonicaClient
import sys
import Ice

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Script description')
    #parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    #parser.add_argument(dest='files', nargs='+')
    parser.add_argument('-t','--test', action='store_true', help='Make test data, not real data', default=False)
    parser.add_argument('--log-config', help='log config')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
  
    logger = logging.getLogger(__file__)
    logging.init_logging(sys.argv)
    communicator = Ice.initialize(sys.argv)

    mon = MonicaClient(('icehost-mro',8051))
    ants = [1,2,3,4,5,6,12,14,16,17,19,24,27,28]
    print('Running moncraft with ', len(ants), 'antennas', ants, 'test?', values.test)

    try:
        craft = MonicaCraftManager(communicator, mon, ants, values)
        craft.run()
        pass
    except Exception as ex:
        logger.error(str(ex))
        print(ex, file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
