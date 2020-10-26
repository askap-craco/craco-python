#!/usr/bin/env python

"""
A wrapper of ETCD3 python package (https://pypi.org/project/etcd3/)

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import argparse
from os import path
import etcd3

class Etcd3Wrapper(object):
    '''
    A class to wrapper useful ETCD3 python functions
    '''

    def __init__(self,
                 server,
                 root):

        # Setup arguments
        self._server = server
        self._root   = root
        
        # Get log setup
        self._log = logging.getLogger(__name__)
        
        self._etcd = etcd3.client(host=self._server.split(":")[0],
                                  port=self._server.split(":")[1])

    def get_keys(self, key_root, keys):
        self._log.info("Parsing '{}' ETCD key values".format(key_root))
        values = {}
        for key in keys:
            try:
                values[key] = self._etcd.get(path.join(self._root,
                                                       "{}/{}".format(key_root,
                                                                      key)))[0].decode('UTF-8')
            except Exception as error:
                self._log.exception(error)
                raise Exception(error)

        return values
    
    def put_keys(self, key_root, key_values):
        '''
        Write key value pairs from key_values dictionary
        to a directory defined by self._root/key_root.
        
        The function assume that key_values is not nested.
        It loops all keys in the key_values dictionary and 
        setup these keys with associate values to the directory.
        
        Args:
        key_root (string):       The root to hold these keys.
        key_values (dictionary): Dictionary to hold key and value pairs
        '''
        try:
            self._log.info("Setting up '{}' key values to ETCD server".format(key_root))
            for key, value in key_values.items():
                try:
                    self._log.debug("Setting up '{}' key value to '{}'".format(key, value))
                    self._etcd.put(path.join(self._root, "{}/".format(key_root)+key), str(value))
                except Exception as error:
                    log.error(error)
                    raise Exception(error)                    
        except:
            raise Exception("{} configuration failed".format(key_root))
