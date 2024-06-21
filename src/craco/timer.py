#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import logging
import time
from collections import OrderedDict
from craco.tracing import tracing

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class Timestamp:
    def __init__(self, perf, process, tai_ns):
        self.perf = perf # *does* include time spent in sleep
        self.process = process # system+CPU time. Does *not* include time spent in sleep
        self.tai_ns = tai_ns

    @staticmethod
    def now():
        return Timestamp(time.perf_counter(), time.process_time(), time.clock_gettime_ns(time.CLOCK_TAI))

    @property
    def sleep(self):
        '''
        Returns the amount of time spent sleeping'''
        s = self.perf - self.process
        return s

    @property
    def walltime(self):
        return self.perf

    def __float__(self):
        '''
        Returns time spentincluding sleep
        '''
        
        return self.perf

    def __sub__(self, ts):
        return Timestamp(self.perf - ts.perf, self.process - ts.process, self.tai_ns - ts.tai_ns)

    def __str__(self):
        s = f'{self.process*1e3:0.1f}ms CPU + {self.sleep*1e3:0.1f}ms sleep'
        return s

class Timer:
    def __init__(self, args=None):
        self.last_ts = Timestamp.now()
        self.init_ts = self.last_ts
        self.ticks = OrderedDict()
        try:
            from craco.mpi_tracefile import MpiTracefile
            self.tracefile = MpiTracefile.instance()
        except:
            log.info('Could not initialise MpiTracefile')
            self.tracefile = None

        self.args = {} if args is None else args

    def tick(self, name, args=None):
        ts = Timestamp.now()
        tdiff = ts - self.last_ts
        self.ticks[name] = tdiff

        # add completion event for this thing
        # timestamps are integer microseconds
        allargs = dict(self.args)
        if args is not None:
            allargs.update(args)

        if self.tracefile is not None:
            complete_event = tracing.CompleteEvent(name, 
                ts = self.last_ts.tai_ns //1e3, 
                dur=int(tdiff.perf*1e6), 
                tdur=int(tdiff.process*1e6),
                args=allargs) # not sure if this should be process or perf?
            self.tracefile.tracefile += complete_event
        
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
