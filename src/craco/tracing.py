#!/usr/bin/env python
"""
Tracing classes. Based loosely on https://github.com/lochbrunner/trace_event_handler/blob/master/trace_event_handler/trace_event_handler.py

Writes google trace event format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview

Copyright (C) CSIRO 2022
"""

import os
import sys
import logging
import json
import logging
import os
import sys
import threading
import time
import traceback

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class TrivialEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

class Event:
    def __init__(self, name, ts, sf=None, ph='B'):
        self.pid = os.getpid()
        self.tid = threading.current_thread().ident
        self.ts = ts
        self.cat = ''
        self.ph = ph
        if type(name) == bytes:
            name = name.decode('utf-8')
        elif type(name) != str:
            name = str(name)
        if len(name) <= 30:
            self.name = name
        else:
            self.name = name[:27] + '...'
        if sf is not None:
            self.sf = sf
        self.args = {}

    def end(self, ts):
        return Event(
            name=self.name,
            ts=ts,
            ph='E',
            sf=self.sf
        )

class Tracefile:
    def __init__(self, fname, type='array'):
        '''
        type='array' is simple. Can't store stack frames
        type='object', more complex. can store stack frames. less robust if it's not closed properly
        '''
        assert type in ('array','object')
        self.type = type
        self.fout = open(fname, 'wt')
        self._nevents = 0
        fout = self.fout
        if type == 'array':
            fout.write('[\n')
        else:
            fout.write('{ "traceEvents": [\n')

    @property
    def stackframes(self):
        return []

    def append(self, d):
        s = json.dumps(d, cls=TrivialEncoder)
        fout = self.fout
        if self._nevents > 0:
            fout.write(',\n')
            
        fout.write(s)
        self._nevents += 1

        return self

    __iadd__ = append

    def close(self):
        fout = self.fout
        if self.type == 'array':
            fout.write(']')
        else:
            fout.write('\n],')
            sf = json.dumps(self.stackframes)
            fout.write(f'"stackFrames":{sf}')
            fout.write('\n}')

        fout.close()
        self.fout = None
        

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
