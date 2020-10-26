#!/usr/bin/env python
import unittest
import coloredlogs
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from craco.pipeline import Pipeline
from craco.execute import ExecuteCommand
from craco.yaml2etcd import Yaml2Etcd

"""
Script to test CRACO pipeline.

Copyright (C) CSIRO 2020
"""

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

# A class to emulation argparse namespace class
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestUdpdb(unittest.TestCase):
    def setUp(self):
        """Call before every test case."""
        pass
   
    def tearDown(self):
        """Call after every test case."""
        pass
   
    def test_udpdb(self):
        '''
        Implementation of the test function
        compare data file from correlator simulator and dbdisk
        '''
        # Setup the test configuration first
        config = Namespace(etcd_server=["localhost:2379"],
                           etcd_root=["/test_udpdb"],
                           yaml_fname=["test_udpdb.yaml"],
                           execution=[False])
    
        # Load ETCD configuration
        yaml2etcd = Yaml2Etcd.from_args(config)
        yaml2etcd.run()

        # Run essential applications to generate files
        pipeline = Pipeline.from_args(config)
        pipeline.run()
        
        # details to check result
        # need to parse ETCD values also

def _main():
    parser = ArgumentParser(description='To test the CRACO pipeline')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')
    parser.set_defaults(verbose=False)    
    values = parser.parse_args()
    
    ## Setup logger
    logging.basicConfig(filename='{}.log'.format(__file__.split("/")[-1].split(".")[0]))
    log = logging.getLogger(__file__.split("/")[-1].split(".")[0])
    if values.verbose:
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='DEBUG')
    else:            
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='INFO')

    # To test for real
    unittest.main() # run all tests

if __name__ == "__main__":
    # ./test_pipeline.py -v
    
    _main()
