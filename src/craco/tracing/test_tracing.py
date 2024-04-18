#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import pytest
from craco.tracing import *
import json

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"
def jsonfile(f):
    with open(f, 'rt') as fin:
        d = json.load(fin)

    return d

test_events = [CompleteEvent('hello', ts=100, dur=100, tdur=50),
                   CompleteEvent('hello2',ts=200,dur=200,tdur=200),
                   CompleteEvent('bye',ts=200,dur=100,tdur=50,tid=1234,pid=5555,cat='good'),
                   CompleteEvent('bye2',ts=400,dur=100,tdur=50,tid=1234,pid=5555,cat='good'),
                   CounterEvent('animals', ts=0,args={'cats':0,'dogs':10}),
                   CounterEvent('animals', ts=10,args={'cats':10,'dogs':30}),
                   CounterEvent('animals', ts=20,args={'cats':20,'dogs':20}),
                   InstantEvent('KB thread event',ts=150,s='t'), # Seems not todisplay in perfetto
                   InstantEvent('KB process event',ts=250,s='p'),
                   InstantEvent('KB global event',ts=350,s='g', cat='global_event'),
                   # Async event example
                   AsyncEvent(ts=100, cat='foo',name='async_read',_id=0x100,ph='b',args={'name':'~/.bashrc'}),
                   AsyncEvent(ts=200, cat='foo', name='async_read',_id=0x100,ph='e'),

                   # Nested async event example
                   AsyncEvent(cat='foo',name='url_request',ph='b',ts=0,_id=0x200),
                   AsyncEvent(cat='foo',name='url_headers',ph='b',ts=100,_id=0x200),
                   AsyncEvent(cat='foo',name='http_cache', ph='n',ts=300,_id=0x200),
                   AsyncEvent(cat='foo',name='url_headers',ph='e',ts=200,_id=0x200,args={'step':'headers_complete','response_code':200}),
                   AsyncEvent(cat='foo',name='url_request',ph='e',ts=400,_id=0x200),

                   # Flow aevent example?
                   FlowEvent(cat='dog',name='url_request2',ph='s',ts=1000,_id=0x300, pid=20),
                   #FlowEvent(cat='dog',name='url_headers2',ph='s',ts=1100,_id=0x300, pid=20),
                   #FlowEvent(cat='dog',name='http_cache2', ph='t',ts=1300,_id=0x300, pid=20),
                   #FlowEvent(cat='dog',name='url_header2',ph='f',ts=1200,_id=0x300,args={'step':'headers_complete','response_code':200}, pid=20),
                   FlowEvent(cat='dog',name='url_request2',ph='f',ts=1400,_id=0x300, pid=20)
                   
                   
                   ]

def check_event_list(l1, l2):
    assert len(l1) == len(l2)
    for d1, d2 in zip(l1,l2):
        for k,v2 in d2.items():
            if k == 'id':
                kread = '_id'
            else:
                kread = k
                
            v1 = getattr(d1,kread)
            assert v1 == v2

def test_tracefile_array_is_json():
    fname = 'trace_array.json'
    t = Tracefile(fname, type='array')
    e = Event('hello', 1)
    t += e
    t.close()
    d = jsonfile(fname)
    check_event_list([e], d)

def test_tracefile_object_is_json():
    fname = 'trace_object.json'
    t = Tracefile(fname, type='object')
    e = Event('hello', 1)
    t += e
    t.close()
    d = jsonfile(fname)
    assert d['stackFrames'] == []
    check_event_list([e],d['traceEvents'])

def test_counter_is_ok():
    c = CounterEvent('ctr',ts=0,args={'cats':0})
    assert c.ph == 'C'

def test_complete_is_ok():
    c = CompleteEvent('bye2',ts=400,dur=100,tdur=50,tid=1234,pid=5555,cat='good')
    assert c.ph == 'X'

def test_tracefile_array_of_tests_json():
    fname = 'trace_test_events.json'
    t = Tracefile(fname, type='array')
    
    for e in test_events:
        t += e
    t.close()
    d = jsonfile(fname)
    check_event_list(test_events, d)
    

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
