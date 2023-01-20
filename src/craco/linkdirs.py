#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import numpy as np
import os
import sys
import logging
import shutil
import glob
from pathlib import Path

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

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

    for d in values.files:
        root = Path(d)
        #root = Path(os.path.basename(d))
        local_root = Path(*root.parts[:-1]) # inc
        for (dirpath, dirnames, filenames) in os.walk(d):
            localdir = Path(dirpath).relative_to(local_root)
            os.makedirs(localdir, exist_ok=True)
            for f in filenames:
                src = os.path.join(dirpath, f)
                dst = os.path.join(localdir, f)
                print('link', src, dst)
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                
    

if __name__ == '__main__':
    _main()
