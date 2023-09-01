#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import logging
import time
from collections import OrderedDict

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Timestamp:
    def __init__(self, perf, process):
        self.perf = perf # does not include time spent in sleep
        self.process = process # includes time spent in sleep

    @staticmethod
    def now():
        return Timestamp(time.perf_counter(), time.process_time())


    @property
    def sleep(self):
        '''
        Returns the amoutn of time spent sleeping'''
        s = self.perf - self.process
        return s

    def __sub__(self, ts):
        return Timestamp(self.perf - ts.perf, self.process - ts.process)

    def __str__(self):
        s = f'{self.process*1e3:0.1f}ms CPU + {self.sleep*1e3:0.1f}ms sleep'
        return s

class Timer:
    def __init__(self):
        self.last_ts = Timestamp.now()
        self.init_ts = self.last_ts
        self.ticks = OrderedDict()

    def tick(self, name):
        ts = Timestamp.now()
        tdiff = ts - self.last_ts
        self.ticks[name] = tdiff
        self.last_ts = ts
        
        return tdiff

    @property
    def total(self):
        '''
        Returns last minus first timestamp
        '''
        t = self.last_ts - self.init_ts
        return t
        

    def __str__(self):
        s = '. '.join([f'{tick[0]}:{tick[1]}' for tick in self.ticks.items()])
        s += f'. Total: {self.total}'
        return s

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
