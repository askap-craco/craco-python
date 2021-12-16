#!/usr/bin/env python
"""
MPI Utility classe

Copyright (C) CSIRO 2020
"""
import mpi4py
import logging

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

log = logging.getLogger(__name__)

class MpiPipeline:
    def __init__(self, nbeam:int):
        self.nbeam = nbeam
        self.__beam_processes = []
        self.__root_proceses = []

    def beam_process(self, func):
        self.__beam_processes.append(func)

    def root_process(self, func):
        self.__root_processes.append(func)


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
