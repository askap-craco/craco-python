#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import logging
import time

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Timestamp:
    def __init__(self, perf, process):
        self.perf = perf
        self.process = process

    @staticmethod
    def now():
        return Timestamp(time.perf_counter(), time.process_time())

    def __sub__(self, ts):
        return Timestamp(self.perf - ts.perf, self.process - ts.process)

    def __str__(self):
        s = f'Perf {self.perf*1e3:0.1f}ms Proc {self.process*1e3:0.1f}ms'
        return s

class Timer:
    def __init__(self):
        self.last_ts = Timestamp.now()
        self.init_ts = self.last_ts
        self.ticks = []

    def tick(self, name):
        ts = Timestamp.now()
        self.ticks.append((name, ts - self.last_ts))
        self.last_ts = ts

    def __str__(self):
        return ' '.join([f'{tick[0]}:{tick[1]}' for tick in self.ticks])
        
        

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
