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
from craco.tracing.tracing import *

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-O','--outfile', default='output.json')
    #parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    t = Tracefile(values.outfile)
    pids = {}
    for line in sys.stdin:
        bits = line.split()
        if len(bits) == 9:
            # e.g.
            #             perf 1575974 [000] 243624.657407:          1   cycles:  ffffffffaf87c7e4 [unknown] ([kernel.kallsyms])

            cmd, pid, cpu, time, cycles, cycle_label, addr, func, lib = bits
            print(bits)
            time = time.replace(':','')
            if pid not in pids.keys():
                pids[pid] = cmd
                t.add_metadata(pid, process_name=cmd)
            
            # timestamp is in microsoeconds according ot this: 
            # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0
            t += CompleteEvent(ts=float(time)*1e6,dur=float(cycles)/1e3,pid=pid, name=func)

    t.close()

        

    

if __name__ == '__main__':
    _main()
