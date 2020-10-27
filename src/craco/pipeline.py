#!/usr/bin/env python

"""
Script to run CRACO pipeline with different mode. 

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import argparse
from os import path
from craco.etcd3 import Etcd3Wrapper
from craco.execute import ExecuteCommand

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

class PipelineError(Exception):
    pass

class Pipeline(object):
    '''
    Class to define all function and behaviours 
    to run pipeline with different configurations
    '''
    def __init__(self, values):
        self._execution   = values.execution
        self._etcd_server = values.etcd_server[0]
        self._etcd_root   = values.etcd_root[0]

        # Get log setup
        self._log = logging.getLogger(__name__)
        #self._log = logging.getLogger(None)

        # Setup ETCD client
        self._log.info("Setting up ETCD client")
        self._etcd = Etcd3Wrapper(self._etcd_server,
                                  self._etcd_root)
                
        # Reset at the very beginning
        self._log.info("Reset at the beginning")
        self._reset()
        
        # Default we do not have ring buffer
        self._has_db = False
        
        # Record all execution instances
        # just in case we need to kill them all
        self._execution_instances = []
        
        # DB part has no seperate function
        # create ring buffer is part of initial
        self._log.info("Parse 'basic' keys from ETCD")
        
        etcd_keys = {"writer", "reader", "key",   "ntime",
                     "nchan",  "nbl",    "npol",   "nbyte",
                     "app", "reader_accessory", "writer_accessory",
                     "tsamp", "centre_freq", "bw"}
        try:
            etcd_values = self._etcd.get_keys("basic", etcd_keys)
        except Exception as error:
            self._log.exception(error)

            self._terminate_executions()
            self._reset()
            
            raise PipelineError(error)
            
        # Not shared with other functions for sure
        app = etcd_values["app"]
        
        # Shared configurations for sure
        self._key    = etcd_values["key"]
        self._writer = etcd_values["writer"]
        self._reader = etcd_values["reader"]
        self._writer_accessory = etcd_values["writer_accessory"]
        self._reader_accessory = etcd_values["reader_accessory"]
        
        nreader = 1 
        if type(self._reader) is list:
            nreader = len(self._reader) # Not shared for sure
        self._log.debug("reader list is {}".format(self._reader))
        self._log.debug("writer list is {}".format(self._writer))
        self._log.debug("reader accessory list is {}".format(self._reader_accessory))
        self._log.debug("writer accessory list is {}".format(self._writer_accessory))
        self._log.debug("We have {} readers".format(nreader))
                
        # May shared 
        self._ntime  = int(etcd_values["ntime"])
        self._nchan  = int(etcd_values["nchan"])
        self._nbl    = int(etcd_values["nbl"])
        self._npol   = int(etcd_values["npol"])        
        self._nbyte  = int(etcd_values["nbyte"])
        
        self._tsamp        = float(etcd_values["tsamp"])
        self._centre_freq  = float(etcd_values["centre_freq"])
        self._bw           = float(etcd_values["bw"])
        
        self._log.debug("ntime is {}".format(self._ntime))
        self._log.debug("nchan is {}".format(self._nchan))
        self._log.debug("nbl is   {}".format(self._nbl))
        self._log.debug("npol is  {}".format(self._npol))
        self._log.debug("tsamp is       {} microsecond".format(self._tsamp))
        self._log.debug("centre_freq is {} MHz".format(self._centre_freq))
        self._log.debug("bw is          {} MHz".format(self._bw))
        
        # build dada_db command line
        blksz = self._npol*self._ntime*self._nbl*self._nchan*self._nbyte
        command = "{} -k {} -r {} -b {}".format(app,
                                                self._key,
                                                nreader, 
                                                blksz)
        self._log.info("Create ring buffer as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        execution_instance.finish()
        
        # Now we have ring buffer
        self._has_db = True

    @classmethod
    def from_args(cls, values):
        '''
        Build a pipeline from command line arguments
        '''
        return Pipeline(values)
    
    def run(self):
        # reader_worker, reader_accessory_worker
        # writer_worker, writer_accessory_worker
        # are order sensitivity
        # We do not initialise all values at the startup, 
        # instead, we do that at seperate functions to
        # make sure that the dependence of these functions is meet
        
        # When the pipeline object initialized,
        # all tasks initialized and wait to finish
                
        # Attach all readers to the ring buffer
        self._run_reader()

        # Attach reader_accessory to the ring buffer
        # We may not have any reader accessory
        if self._reader_accessory:
            self._run_reader_accessory()
        
        # Attach writer to the ring buffer
        self._run_writer()

        # Attach writer_accessory to the ring buffer
        # We may not have any writer accessory
        if self._writer_accessory:
            self._run_writer_accessory()
        
        # Wait all applications finish
        self._sync_executions()
        
    def _run_reader(self):
        if "search" in self._reader:
            self._search()
        if "dbdisk" in self._reader:
            self._dbdisk()            
               
    def _run_reader_accessory(self):        
        if "average" in self._reader_accessory:
            self._average()
        if "uvgrid" in self._reader_accessory:
            self._uvgrid()
        if "calibration" in self._reader_accessory:
            self._calibration()

    def _run_writer(self):
        if "diskdb" in self._writer:
            self._diskdb()
        if "udpdb" in self._writer:
            self._udpdb()
        if "simulator" in self._writer:
            self._simulator()
        if "correlator" in self._writer:
            self._correlator()
        
    def _run_writer_accessory(self):
        pass

    def _diskdb(self):
        self._log.info("Parse 'diskdb' keys from ETCD")
        etcd_keys = {"app", "file_name"}
        try:
            etcd_values = self._etcd.get_keys("diskdb", etcd_keys)
        except Exception as error:
            self._log.exception(error)

            self._terminate_executions()
            self._reset()
            
            raise PipelineError(error)
        
        app       = etcd_values["app"]
        file_name = etcd_values["file_name"]

        # Build diskdb command line
        command = "{} -k {} -f {}".format(app,
                                          self._key,
                                          file_fname)
        
        self._log.info("Run diskdb as '{}'".format(diskdb))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        # Do not block it

    def _simulator(self):
        command = ("corrsim  /data/craco/ban115/test_data/"
                   "frb_d0_t0_a1_sninf_lm00/frb_d0_t0_a1_sninf_lm00.fits"
                   " -d localhost:10174 -b 3 -p 0,1 --nloops 100")
        
        self._log.info("Run simulator as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _correlator(self):
        '''
         Place holder for correlator function
         We may not be able to control correlator anyway
        '''
        pass

    def _calibration(self):
        '''
        Place holder for calibration function.
        It will be designed for end-to-end pipeline.

        It receives average data from ZeroMQ,
        calculates calibration and sends it to a different ZeroMQ.
        '''
        pass

    def _uvgrid(self):
        '''
        Place holder for uvgrid function.
        It will be designed for end-to-end pipeline.

        It sends uvgrid to a ZeroMQ.
        '''
        pass

    def _uvgrid_receiver(self):
        '''
        place holder for uvgrid receiver function.
        It will be designed for test. 

        It receives uvgrid data from a ZeroMQ interface and 
        compare to something.
        '''
        pass
    
    def _trigger(self):
        '''
        Place holder for trigger function.
        It will be designed for end-to-end pipeline.

        It receives candidate from 'search' application and 
        does trigger thing.
        '''
        pass
    
    def _average(self):
        '''
        Place holder for average function
        It will be designed to do tests.
        can also be used as part of end-to-end pipeline.

        It receives average data from ZeroMQ and
        compares it with something
        '''
        pass

    def _candidate(self):
        '''
        Place holder for average function
        It will be designed to do tests.
        Can also be used as part of end-to-end pipeline.

        It receives candidates from ZeroMQ and
        compares them with something
        '''
        pass

    def _udpdb(self):
        # Build udpdb command line
        self._log.info("Parse 'udpdb' keys from ETCD")
        etcd_keys = {"app", "udpdb_zmq"}
        try:
            etcd_values = self._etcd.get_keys("udpdb", etcd_keys)
        except Exception as error:
            self._log.exception(error)

            self._terminate_executions()
            self._reset()
            
            raise PipelineError(error)

        # Not shared for sure
        app = etcd_values["app"]

        # Shared with its accessory
        self._udpdb_zmq = etcd_values["udpdb_zmq"]
        
        command = "{} -a {} -b {}".format(app,
                                          self._etcd_server,
                                          self._etcd_root)
        self._log.info("Run udpdb as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _dbdisk(self):
        self._log.info("Parse 'dbdisk' keys from ETCD")
        etcd_keys = {"app", "directory"}
        try:
            etcd_values = self._etcd.get_keys("dbdisk", etcd_keys)
        except Exception as error:
            self._log.exception(error)

            self._terminate_executions()
            self._reset()
            
            raise PipelineError(error)

        app       = etcd_values["app"]
        directory = etcd_values["directory"]

        # Build dbdisk command line
        command = "{} -k {} -D {}".format(app,
                                          self._key,
                                          directory)
        
        self._log.info("Run dbdisk as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _search(self):        
        self._log.info("Parse 'search' keys from ETCD")
        etcd_keys = {"app",         "uvgrid_zmq", "calibration_zmq",
                     "average_zmq", "candidate_zmq"}
        try:
            etcd_values = self._etcd.get_keys("search", etcd_keys)
        except Exception as error:
            self._log.exception(error)

            self._terminate_executions()
            self._reset()
            
            raise PipelineError(error)

        # Not shared for sure
        app = etcd_values["app"]

        # May shared
        self._uvgrid_zmq      = etcd_values["uvgrid_zmq"]
        self._calibration_zmq = etcd_values["calibration_zmq"]
        self._average_zmq     = etcd_values["average_zmq"]
        self._candidate_zmq   = etcd_values["candidate_zmq"]
        
        # Build search command line
        command = "{} -a {} -b {}".format(app,
                                          self._etcd_server,
                                          self._etcd_root)
        self._log.info("Run search as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _sync_executions(self):
        # Wait all executions finish
        self._log.info("To check if we have launch failure")
        failed_launch = False
        for execution_instance in self._execution_instances:
            failed_launch = (failed_launch or execution_instance.failed_launch)

        if failed_launch:
            self._log.info("Have to terminate all applications "
                     "as launch failure happened")
            self._terminate_executions()
        else:
            self._log.info("Wait all executions to finish")
            for execution_instance in self._execution_instances:
                execution_instance.finish()
            
        if(self._has_db):
            self._log.info("Destroy ring buffer '{}'".format(self._key))
            command = ("dada_db -d -k {} ").format(self._key)
            execution_instance = ExecuteCommand(command,
                                                self._execution)
            self._execution_instances.append(execution_instance)
            execution_instance.returncode_callbacks.add(self._returncode_handle)
            execution_instance.stdout_callbacks.add(self._stdout_handle)
            execution_instance.finish()

        self._log.info("Reset everything at the end")
        self._reset()

    def _reset(self):
        commands = ["ipcrm -a",
                    "pkill -9 -f yaml2etcd",
                    "pkill -9 -f corrsim",
                    "pkill -9 -f udpdb",
                    "pkill -9 -f search",
                    "pkill -9 -f dada_db",
                    "pkill -9 -f dada_diskdb",
                    "pkill -9 -f dada_dbdisk"
                    #"xbutil reset"
        ]
        execution_instances = []
        for command in commands:
            #print (command)
            self._log.debug("cleanup with {}".format(command))
            execution_instances.append(
                ExecuteCommand(command, self._execution, "y"))
        for execution_instance in execution_instances:
            # Wait until the reset is done
            execution_instance.finish()

        # ring buffer is deleted by now
        self._has_db = False
        
    def _returncode_handle(self, returncode, callback):
        # Rely on stdout_handle to print all information
        if self._execution:
            if returncode:
                self._log.error(returncode)
                
                self._log.error("Terminate all execution instances "
                          "when error happens")
                self._terminate_executions()
                
                self._log.error("Reset when error happens")
                self._reset()
                
                raise PipelineError(returncode)

    def _stdout_handle(self, stdout, callback):
        if self._execution:
            self._log.debug(stdout)

    def _terminate_executions(self):
        for execution_instance in self._execution_instances:
            execution_instance.terminate()

def _main():
    parser = argparse.ArgumentParser(
        description='To run the CRACO pipeline')
    parser.add_argument('-s', '--etcd_server', type=str, nargs='+',
                        help='ETCD server to configure the pipeline')
    parser.add_argument('-r', '--etcd_root', type=str, nargs='+',
                        help='ETCD root to configure the pipeline')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')
    parser.add_argument('-e', '--execution', action='store_true',
                        help='Execution or not')
    parser.set_defaults(execution=False)
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

    pipeline = Pipeline.from_args(values)
    pipeline.run()

if __name__ == "__main__":
    # pipeline -s localhost:2379 -r /SB123/beam01 -v -e
    _main()
