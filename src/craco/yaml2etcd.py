#!/usr/bin/env python
"""
Script to load key value from a YAML file to the given root of an ETCD server.

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
from os import path
from craco.etcd3 import Etcd3Wrapper
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
__author__ = "Xinping Deng <xinping.deng@csiro.au>"

class Yaml2Etcd():
    def __init__(self, values):
        # Get log setup
        self._log = logging.getLogger(__name__)
        #self._log = logging.getLogger(None)
        
        # Get values from outside first
        self._etcd_root   = values.etcd_root[0]
        self._etcd_server = values.etcd_server[0]
        yaml_fname        = values.yaml_fname[0]
        
        # Open YAML file and load data to memory
        try:
            with open(yaml_fname) as f:
                self._yaml_dict = yaml.load(f, yaml.Loader)
        except:
            raise Exception("Can not open yaml file '{}'".format(yaml_fname))
        
        # Setup ETCD client
        self._log.info("Setting up ETCD client")
        #host = self._etcd_server.split(":")[0]
        #port = self._etcd_server.split(":")[1]
        self._log.debug("ETCD server is running on "
                        "{}".format(self._etcd_server))
        try:
            #self._etcd = etcd3.client(host=host, port=port)
            self._etcd = Etcd3Wrapper(self._etcd_server, self._etcd_root)
        except:
            raise Exception("Can not create etcd client with "
                            "'{}'".format(self._etcd_server))

    @classmethod
    def from_args(cls, values):
        '''
        Build a Yaml2Etcd from command line arguments
        '''
        return Yaml2Etcd(values)

    def run(self):
        # Check and load basic configuration
        try:
            basic = self._yaml_dict["basic"]
            self._log.info("Loading 'basic' key values to ETCD server")
            self._etcd.put_keys("basic",
                                self._yaml_dict["basic"])
        except:
            self._log.error("'basic' configuration failed")
            raise Exception("'basic' configuration failed")
    
        # Setup key words for all readers
        try:
            reader = basic["reader"]
            if " " not in reader:
                self._etcd.put_keys(reader,
                                    self._yaml_dict[reader])
            else:
                reader_parts = reader.split(" ")
                for reader_part in reader_parts:
                    self._etcd.put_keys(reader_part,
                                        self._yaml_dict[reader_part])
        except:
            self._log.error("Failed to setup 'reader' from 'basic' section")
            raise Exception("Failed to setup 'reader' from 'basic' section")

        # Setup key words for all reader accessory applications
        try:
            reader_accessory = basic["reader_accessory"]
            if reader_accessory != None:
                if " " not in reader_accessory:
                    self._etcd.put_keys(reader_accessory,
                                        self._yaml_dict[reader_accessory])
                else:
                    reader_accessory_parts = reader_accessory.split(" ")
                    for reader_accessory_part in reader_accessory_parts:
                        self._etcd.put_keys(reader_accessory_part,
                                            self._yaml_dict[reader_accessory_part])
        except:
            self._log.error("Failed to setup 'reader_accessory' from 'basic' section")
            raise Exception("Failed to setup 'reader_accessory' from 'basic' section")

        # Check and load writer key words
        try:
            writer = basic["writer"]
            if "_" not in writer:
                self._etcd.put_keys(writer,
                                    self._yaml_dict[writer])
            else:
                writer_parts = writer.split("_")
                for writer_part in writer_parts:
                    self._etcd.put_keys(writer_part,
                                        self._yaml_dict[writer_part])
        except:
            self._log.error("Failed to setup 'writer' from 'basic' section")
            raise Exception("Failed to setup 'writer' from 'basic' section")

        # Setup key words for all writer accessory applications
        try:
            writer_accessory = basic["writer_accessory"]
            if writer_accessory != None:
                if " " not in writer_accessory:
                    self._etcd.put_keys(writer_accessory,
                                        self._yaml_dict[writer_accessory])
                else:
                    writer_accessory_parts = writer_accessory.split(" ")
                    for writer_accessory_part in writer_accessory_parts:
                        self._etcd.put_keys(writer_accessory_part,
                                            self._yaml_dict[writer_accessory_part])
        except:
            self._log.error("Failed to setup 'writer_accessory' from 'basic' section")
            raise Exception("Failed to setup 'writer_accessory' from 'basic' section")
    
def _main():
    parser = ArgumentParser(description='Load ETCD configurations from a YAML file', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--etcd_server', type=str, nargs='+',
                        help='ETCD server to take configuration')
    parser.add_argument('-r', '--etcd_root', type=str, nargs='+',
                        help='ETCD root to take configuration')
    parser.add_argument('-f', '--yaml_fname', type=str, nargs='+',
                        help='YAML file to load configuration')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    ## Setup logger
    logging.basicConfig(filename='{}.log'.format(__name__))
    log = logging.getLogger(__name__)
    if values.verbose:
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='DEBUG')
    else:            
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='INFO')

    yaml2etcd = Yaml2Etcd.from_args(values)
    yaml2etcd.run()
    
if __name__ == '__main__':
    # yaml2etcd -s localhost:2379 -r /SB123/beam01 -f ../../etcd/etcd_template.yaml -v
    _main()
