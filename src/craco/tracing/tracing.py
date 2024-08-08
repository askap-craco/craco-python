#!/usr/bin/env python
"""
Tracing classes. Based loosely on https://github.com/lochbrunner/trace_event_handler/blob/master/trace_event_handler/trace_event_handler.py

Writes google trace event format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview

Copyright (C) CSIRO 2024
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
import numpy as np

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

DEPRECATED_PHASES = 'ISTPF'

ALL_PHASES = 'BEXiCbnestfPNODMVvRc'

class TrivialEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, '__dict__'):
            d = o.__dict__.copy()

            # id can't be an attribute, so we stire it as _id
            # as an attribute and convert to the non underscored version in the encoder
            if '_id' in d:
                d['id'] = d['_id']
                del d['_id']
        elif np.issubdtype(o, np.integer): #https://numpy.org/devdocs/reference/generated/numpy.issubdtype.html
            d = int(o)            
        elif np.issubdtype(o, np.floating):
            d = float(o)        
        else:
            raise ValueError(f'Cant encode {o} {type(o)}')

        return d

class Event:
    def __init__(self, name, ts, sf=None, ph='B', pid=None, tid=None, cat='', args=None, tts=None):
        #print(self, name, ts, sf, ph)
        if pid is None:
            pid = os.getpid()
        if tid is None:
            tid = threading.current_thread().ident

        assert ph in ALL_PHASES, f'Invalid ph={ph} not in {ALL_PHASES}'

        if pid is not None:
            self.pid = pid

        if tid is not None:
            self.tid = tid

        if ts is None:
            ts = time.clock_gettime(time.CLOCK_TAI)*1e6

        self.ts = ts    
        
        if cat is None:
            cat = ''

        self.cat = cat
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

        if args is None:
            self.args = {}
        else:
            self.args = args

        if tts is not None:
            self.tts = tts

    def end(self, ts):
        return Event(
            name=self.name,
            ts=ts,
            ph='E',
            sf=self.sf
        )

class CompleteEvent(Event):
    def __init__(self, name, ts, dur, tdur=None, sf=None, pid=None, tid=None, cat='', args=None):
        super().__init__(name, ts, sf, 'X', pid, tid, cat, args)
        self.dur = dur
        if tdur is not None:
            self.tdur = tdur

class MetadataEvent(Event):
    def __init__(self, name, pid=None, tid=None, cat='', args=None):
        super().__init__(name, None, None, 'M', pid, tid, cat, args)
        
class CounterEvent(Event):
    def __init__(self, name, ts ,args, _id=None, sf=None, pid=None, tid=None, cat=None):
        super().__init__(name, ts, sf, 'C', pid, tid, cat, args)
        # Hmm - hard to add "id" as it clashes with a funciton name
        if _id is not None:
            self._id = _id


class InstantEvent(Event):
    def __init__(self, name, ts, s=None, sf=None, args=None, pid=None, tid=None, cat=None):
        '''
        :s: Scope - must be 'g','p,'t' = global, process or thread
        '''
        super().__init__(name, ts, sf, 'i', pid, tid, cat, args)

        if s is not None:
            assert s in 'gpt'
            self.s = s

class AsyncEvent(Event):
    def __init__(self, name, ts, ph, _id, scope=None, args=None, sf=None, pid=None, tid=None, cat=None):
        assert ph in 'bne', 'Unknown async event phase'
        super().__init__(name, ts, sf, ph, pid, tid, cat, args)
        assert _id is not None
        self._id = _id

        if scope is not None:
            self.scope = scope

class FlowEvent(Event):
    def __init__(self, name, ts, ph, _id, scope=None, args=None, sf=None, pid=None, tid=None, cat=None):
        assert ph in 'stf', f'Unknown flow event phase:{ph}'
        super().__init__(name, ts, sf, ph, pid, tid, cat, args)
        assert _id is not None
        self._id = _id

        if scope is not None:
            self.scope = scope

    
   

class Tracefile:
    def __init__(self, fname, type='array'):
        '''
        type='array' is simple. Can't store stack frames
        type='object', more complex. can store stack frames. less robust if it's not closed properly
        '''
        assert type in ('array','object')
        self.type = type
        #assert not os.path.exists(fname), f'Tracefile {fname} already exists'
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
    
    def add_metadata(self, 
                     pid=None,tid=None,
                     process_name=None,
                     process_labels=None,
                     process_sort_index=None,
                     thread_name=None,
                     thread_sort_index=None):
        if process_name is not None:
            self += MetadataEvent('process_name', args={'name':process_name}, pid=pid, tid=tid)
        
        if process_labels is not None:
            self += MetadataEvent('process_labels', args={'labels':process_labels}, pid=pid, tid=tid)

        if process_sort_index is not None:
            self += MetadataEvent('process_sort_index', args={'sort_index':process_sort_index}, pid=pid, tid=tid)

        if thread_name is not None:
            self += MetadataEvent('thread_name', args={'name':process_sort_index}, pid=pid, tid=tid)

        if thread_sort_index is not None:
            self += MetadataEvent('thread_sort_index', args={'sort_index':process_sort_index}, pid=pid, tid=tid)
        

    def now_ts(self):
        '''
        Return now in TAI in microseconds as an integer
        Useful if you want to send multiple events through with exactly the same timestamp 
        '''
        ts = int(time.clock_gettime_ns(time.CLOCK_TAI)//1000)
        return ts

    def flush(self):
        self.fout.flush()

    def append(self, d):
        s = json.dumps(d, cls=TrivialEncoder)
        fout = self.fout
        if self._nevents > 0:
            fout.write(',\n')
            
        fout.write(s)
        self._nevents += 1
        #fout.flush()

        return self

    __iadd__ = append

    def close(self):
        if self.fout is None:
            return
        
        fout = self.fout
        if self.type == 'array':
            fout.write('\n]')
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
