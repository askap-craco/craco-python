#!/usr/bin/env python
"""
Script to load key value from a YAML file to the given root of an ETCD server.

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import argparse
from os import path
import etcd3
import yaml

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

def setup_key_values(etcd_root, yaml_dict, key_root):
    '''
    Write key value to a directory defined by etcd_root/key_root.
    The function assume that the yaml_dict is not nested.
    It loops all keys in the yaml dictionary and 
    setup these keys with associate values to the directory.
    
    Args:
    etcd_root (string):     The root directory.
    yaml_dict (dictionary): The dictionary to hold all key-value pairs.
    key_root  (string):     The root to hold these keys.
    '''
    try:
        key_values = yaml_dict[key_root]
        log.info("Setting up '{}' key values to ETCD server".format(key_root))
        for key, value in key_values.items():
            log.debug("Setting up '{}' key value to '{}'".format(key, value))
            etcd.put(path.join(etcd_root, "{}/".format(key_root)+key), str(value))
    except:
        raise Exception("{} configuration failed".format(key_root))

if __name__ == "__main__":
    '''
    A script to setup etcd key value database with information from a yaml file.

    Run example:
    ./yaml2etcd.py -a localhost:2379 -b /SB123/beam01 -c ../../etcd/etcd_template.yaml 
    
    The script only verify that all required configuration sections are there.
    It does not go inside each section to check the details there. 
    User script like pipeline.py should check that. 
    
    '''
    
    logging.basicConfig(filename='craco_etcd_import.log')
    log = logging.getLogger('etcd_import')
    coloredlogs.install(
        fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level='DEBUG',
        # level='INFO',
        logger=log)

    parser = argparse.ArgumentParser(
        description='To import configuration from a given YAML file to a ETCD server with a root')
    parser.add_argument('-a', '--etcd_server', type=str, nargs='+',
                        help='ETCD server to take configuration')
    parser.add_argument('-b', '--etcd_root', type=str, nargs='+',
                        help='ETCD root to take configuration')
    parser.add_argument('-c', '--yaml_fname', type=str, nargs='+',
                        help='YAML file to load configuration')

    # Parse top-level arguments
    args = parser.parse_args()
    etcd_server = args.etcd_server[0]
    etcd_root   = args.etcd_root[0]
    yaml_fname  = args.yaml_fname[0]
    log.info("ETCD server is {}".format(etcd_server))
    log.info("ETCD root is {}".format(etcd_root))
    log.info("YAML file is {}".format(yaml_fname))
    
    # To setup ETCD client
    log.info("Setting up ETCD client")
    host = etcd_server.split(":")[0]
    port = etcd_server.split(":")[1]
    try:
        etcd = etcd3.client(host=host, port=port)
    except:
        raise Exception("Can not create etcd client with '{}'".format(etcd_server))

    # Open YAML file and load data in
    try:
        with open(yaml_fname) as f:
            yaml_dict = yaml.load(f, yaml.Loader)
    except:
        raise Exception("Can not open yaml file '{}'".format(yaml_fname))

    # Check and load basic configuration
    try:
        basic = yaml_dict["basic"]
        log.info("Loading 'basic' key values to ETCD server")
        setup_key_values(etcd_root, yaml_dict, "basic")
    except:
        log.error("'basic' configuration failed")
        raise Exception("'basic' configuration failed")
    
    # Setup key words for all readers
    try:
        reader = basic["reader"]
        if " " not in reader:
            setup_key_values(etcd_root, yaml_dict, reader)
        else:
            reader_parts = reader.split(" ")
            for reader_part in reader_parts:
                setup_key_values(etcd_root, yaml_dict, reader_part)
    except:
        log.error("Failed to setup 'reader' from 'basic' section")
        raise Exception("Failed to setup 'reader' from 'basic' section")

    # Setup key words for all reader accessory applications
    try:
        reader_accessory = basic["reader_accessory"]
        if reader_accessory != None:
            if " " not in reader_accessory:
                setup_key_values(etcd_root, yaml_dict, reader_accessory)
            else:
                reader_accessory_parts = reader_accessory.split(" ")
                for reader_accessory_part in reader_accessory_parts:
                    setup_key_values(etcd_root, yaml_dict, reader_accessory_part)
    except:
        log.error("Failed to setup 'reader_accessory' from 'basic' section")
        raise Exception("Failed to setup 'reader_accessory' from 'basic' section")

    # Check and load writer key words
    try:
        writer = basic["writer"]
        if "_" not in writer:
            setup_key_values(etcd_root, yaml_dict, writer)
        else:
            writer_parts = writer.split("_")
            for writer_part in writer_parts:
                setup_key_values(etcd_root, yaml_dict, writer_part)
    except:
        log.error("Failed to setup 'writer' from 'basic' section")
        raise Exception("Failed to setup 'writer' from 'basic' section")

    # Setup key words for all writer accessory applications
    try:
        writer_accessory = basic["writer_accessory"]
        if writer_accessory != None:
            if " " not in writer_accessory:
                setup_key_values(etcd_root, yaml_dict, writer_accessory)
            else:
                writer_accessory_parts = writer_accessory.split(" ")
                for writer_accessory_part in writer_accessory_parts:
                    setup_key_values(etcd_root, yaml_dict, writer_accessory_part)
    except:
        log.error("Failed to setup 'writer_accessory' from 'basic' section")
        raise Exception("Failed to setup 'writer_accessory' from 'basic' section")
