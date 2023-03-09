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
from craco.datadirs import DataDirs
import glob

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

resultdir = os.environ['CRACO_RESULTS']
assert resultdir.strip() != '', 'Set CRACO_RESULTS environtment variable'
dirs = DataDirs()

def mylink(src, dst):
    log.debug('linking %s->%s', src, dst)
    try:
        os.symlink(src, dst)
    except FileExistsError:
        log.debug('Link %s->%s exists. continuing', src, dst)


def link_ccap_files(sbname, scandir):
    targdir = os.path.join(resultdir, sbname, scandir, 'files')
    log.debug('Making target dir %s', targdir)
    os.makedirs(targdir, exist_ok=True)
    for nd in dirs.node_dirs:
        for infile in glob.glob(os.path.join(nd, sbname, scandir, '*.fits')):
            targfile = os.path.join(targdir, os.path.basename(infile))
            mylink(infile, targdir)

def link_node_dirs(sbname, scandir):
    node_link_dir = os.path.join(resultdir, sbname, scandir, 'nodes')
    log.debug('Making target dir %s', node_link_dir)
    os.makedirs(node_link_dir, exist_ok=True)
    for node_name, node_dir in zip(dirs.node_names, dirs.node_dirs):
        srcdir = os.path.join(node_dir, sbname, scandir)
        dest = os.path.join(node_link_dir, node_name)
        mylink(srcdir, dest)


def prepsb(sbname):
    for scandir in dirs.sb_scan_dumps(sbname):
        #link_ccap_files(sbname, scandir) # takes ages
        link_node_dirs(sbname, scandir)
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Prepare a SB results dir for processing', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    for sb in values.files:
        prepsb(sb)
    

if __name__ == '__main__':
    _main()
