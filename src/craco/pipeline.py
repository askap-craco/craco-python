#!/usr/bin/env python

"""
Script to run CRACO pipeline with different mode. 

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import shlex
from subprocess import PIPE, Popen, check_output
import argparse
import threading
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
from os import path
import etcd3

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

class ExecuteCommand(object):
    '''
    A class to execute command line as a string. 
    It monitor stdout, stderr and returncode with callback functions.
    It also logs all stdout and stderr with debug level.    
    '''
    def __init__(self,
                 command,
                 execution     = True,
                 popup         = None,
                 process_index = None):

        # stdout here includes stdout and stderr
        # some applications do use stderr as stdout
        self.stdout_callbacks     = set()
        self.returncode_callbacks = set()
        
        self.failed_launch = False
        self._stdout       = None
        self._returncode   = None
        
        # Command line
        self._command = command
        self._executable_command = shlex.split(self._command)

        # Useful when we need given options
        self._popup = popup
        
        # To see if we execution the command for real
        self._execution = execution

        # Useful when we have multi pipelines running in parallel
        self._process_index = process_index

        # Setup monitor threads 
        self._monitor_threads = []

        # Setup for force termination of executable
        self._terminate_event = threading.Event()
        self._terminate_event.clear()

        self._process = None
        if self._execution:
            try:
                self._process = Popen(self._executable_command,
                                      stdout=PIPE,
                                      stderr=PIPE,
                                      stdin=PIPE,
                                      bufsize=1,
                                      universal_newlines=True)
                
                if(self._popup): # To write pop up information to the pipe
                    self._process.communicate(input=self._popup)[0]
                else:
                    flags = fcntl(self._process.stdout, F_GETFL)  # Noblock
                    fcntl(self._process.stdout, F_SETFL, flags | O_NONBLOCK)
                    flags = fcntl(self._process.stderr, F_GETFL)
                    fcntl(self._process.stderr, F_SETFL, flags | O_NONBLOCK)
                
            except Exception as error:
                self.failed_launch = True
                self.returncode = self._command + "; RETURNCODE is: ' 1'"
                log.exception("Error while launching command: "
                              "{} with error "
                              "{}".format(self._command, error)) 
                
            # Start monitors
            self._monitor_threads.append(
                threading.Thread(target=self._process_monitor))

            for thread in self._monitor_threads:
                thread.start()

    def __del__(self):
        class_name = self.__class__.__name__

    def finish(self):
        if self._execution:
            for thread in self._monitor_threads:
                thread.join()

    def terminate(self):
        if self._execution:
            if self._process != None and \
               self._process.poll() == None:
                self._terminate_event.set()
                self._process.terminate()

    def stdout_notify(self):
        for callback in self.stdout_callbacks:
            callback(self._stdout, self)

    @property
    def stdout(self):
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        self._stdout = value
        self.stdout_notify()

    def returncode_notify(self):
        for callback in self.returncode_callbacks:
            callback(self._returncode, self)

    @property
    def returncode(self):
        return self._returncode

    @returncode.setter
    def returncode(self, value):
        self._returncode = value
        self.returncode_notify()

    def _process_monitor(self):
        if self._execution:
            while (self._process != None and \
                   self._process.poll() == None) and \
                  (not self._terminate_event.is_set()):
                try:
                    stdout = self._process.stdout.readline().rstrip("\n\r")
                    if stdout != "":
                        if self._process_index != None:
                            self.stdout = "'" + self._command + "' " +\
                                          stdout + \
                                          "; PROCESS_INDEX is " + \
                                          str(self._process_index)
                        else:
                            self.stdout = "'" + self._command + "' " + stdout
                except:
                    pass

                try:
                    stderr = self._process.stderr.readline().rstrip("\n\r")
                    if stderr != "":
                        if self._process_index != None:
                            self.stdout = "'" + self._command + "' " +\
                                          stderr + \
                                          "; PROCESS_INDEX is " + \
                                          str(self._process_index)
                        else:
                            self.stdout = "'" + self._command + "' " + stderr
                except:
                    pass

            if self._process != None and \
               self._process.returncode and \
               (not self._terminate_event.is_set()):
                self.returncode = "'" + self._command + "' " +\
                    "; RETURNCODE is here: " +\
                    str(self._process.returncode)
class PipelineError(Exception):
    pass

class Pipeline(object):
    '''
    Class to define all function and behaviours 
    to run pipeline with different configurations
    '''
    def __init__(self,
                 execution   = True,
                 etcd_server = None,
                 etcd_root   = None):
        self._execution   = execution
        self._etcd_server = etcd_server
        self._etcd_root   = etcd_root

        # Setup ETCD client
        log.info("Setting up ETCD client")
        self._etcd = etcd3.client(host=self._etcd_server.split(":")[0],
                                  port=self._etcd_server.split(":")[1])
        
        # Reset at the very beginning
        log.info("Reset at the beginning")
        self._reset()
        
        # Default we do not have ring buffer
        self._has_db = False
        
        # Record all execution instances
        # just in case we need to kill them all
        self._execution_instances = []

        # DB part has no seperate function
        # create ring buffer is part of initial
        log.info("Parse 'basic' keys from ETCD")
        
        etcd_keys = {"writer", "reader", "key",   "ntime",
                     "nchan",  "nbl",    "npol",   "nbyte",
                     "app", "reader_accessory", "writer_accessory"}

        etcd_values = self._etcd_get_keys("basic", etcd_keys)

        # Shared configurations, part of basic section in ETCD
        self._key    = etcd_values["key"]
        self._writer = etcd_values["writer"]
        self._reader = etcd_values["reader"].split(" ")
        self._writer_accessory = etcd_values["writer_accessory"].split(" ")
        self._reader_accessory = etcd_values["reader_accessory"].split(" ")
        
        nreader = len(self._reader)
        log.debug("reader list is {}".format(self._reader))
        log.debug("We have {} readers".format(nreader))

        ntime  = int(etcd_values["ntime"])
        nchan  = int(etcd_values["nchan"])
        nbl    = int(etcd_values["nbl"])
        npol   = int(etcd_values["npol"])        
        nbyte  = int(etcd_values["nbyte"])
        app    = etcd_values["app"]
        
        log.debug("ntime is {}".format(ntime))
        log.debug("nchan is {}".format(nchan))
        log.debug("nbl is   {}".format(nbl))
        log.debug("npol is  {}".format(npol))
        
        # build dada_db command line
        blksz = npol*ntime*nbl*nchan*nbyte
        command = "{} -k {} -r {} -b {}".format(app,
                                                self._key,
                                                nreader, 
                                                blksz)
        log.info("Create ring buffer as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        execution_instance.finish()
        
        # Now we have ring buffer
        self._has_db = True

        # reader_worker, reader_accessory_worker
        # writer_worker, writer_accessory_worker
        # are order sensitivity
        # We do not initialise all values at the startup, 
        # instead, we do that at seperate functions to enable
        # make sure that the dependence of these functions is meet
        
        # When the pipeline object initialized,
        # all tasks initialized and wait to finish
                
        # Attach all readers to the ring buffer
        self._reader_worker()

        # Attach reader_accessory to the ring buffer
        self._reader_accessory_worker()
        
        # Attach writer to the ring buffer
        self._writer_worker()

        # Attach writer_accessory to the ring buffer
        self._writer_accessory_worker()
        
        # Wait all applications finish
        self._sync_executions()

    def _reader_worker(self):
        for r in self._reader:
            if r == "search":
                self._search()
            if r == "dbdisk":
                self._dbdisk()
                
    def _reader_accessory_worker(self):
        for a in self._reader_accessory:
            if a == "average":
                self._average()
            if a == "uvgrid":
                self._uvgrid()
            if a == "calibration":
                self._calibration()

    def _writer_worker(self):        
        if self._writer == "diskdb":
            self._diskdb()
        else:
            if "simulator" in self._writer:
                self._udpdb()
                self._simulator()
            if "correlator" in self._writer:
                self._udpdb()
                self._correlator()
        
    def _writer_accessory_worker(self):
        pass
                
    def _diskdb(self):
        log.info("Parse 'diskdb' keys from ETCD")
        etcd_keys   = {"app", "file_name"}
        etcd_values = self._etcd_get_keys("diskdb", etcd_keys)

        app       = etcd_values["app"]
        file_name = etcd_values["file_name"]

        # Build diskdb command line
        command = "{} -k {} -f {}".format(app,
                                          self._key,
                                          file_fname)
        
        log.info("Run diskdb as '{}'".format(diskdb))
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
        
        log.info("Run simulator as '{}'".format(command))
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
        log.info("Parse 'udpdb' keys from ETCD")
        etcd_keys   = {"app"}
        etcd_values = self._etcd_get_keys("udpdb", etcd_keys)

        app = etcd_values["app"]
        
        command = "{} -a {} -b {}".format(app,
                                          self._etcd_server,
                                          self._etcd_root)
        log.info("Run udpdb as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
            
    def _dbdisk(self):
        log.info("Parse 'dbdisk' keys from ETCD")
        etcd_keys   = {"app", "directory"}
        etcd_values = self._etcd_get_keys("dbdisk", etcd_keys)

        app       = etcd_values["app"]
        directory = etcd_values["directory"]

        # Build dbdisk command line
        command = "{} -k {} -D {}".format(app,
                                          self._key,
                                          directory)
        
        log.info("Run dbdisk as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _search(self):        
        log.info("Parse 'search' keys from ETCD")
        etcd_keys   = {"app"}
        etcd_values = self._etcd_get_keys("search", etcd_keys)

        app = etcd_values["app"]
        
        # Build search command line
        command = "{} -a {} -b {}".format(app,
                                          self._etcd_server,
                                          self._etcd_root)
        log.info("Run search as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _sync_executions(self):
        # Wait all executions finish
        log.info("To check if we have launch failure")
        failed_launch = False
        for execution_instance in self._execution_instances:
            failed_launch = (failed_launch or execution_instance.failed_launch)

        if failed_launch:
            log.info("Have to terminate all applications "
                     "as launch failure happened")
            self._terminate_executions()
        else:
            log.info("Wait all executions to finish")
            for execution_instance in self._execution_instances:
                execution_instance.finish()
            
        if(self._has_db):
            log.info("Destroy ring buffer '{}'".format(self._key))
            command = ("dada_db -d -k {} ").format(self._key)
            execution_instance = ExecuteCommand(command,
                                                self._execution)
            self._execution_instances.append(execution_instance)
            execution_instance.returncode_callbacks.add(self._returncode_handle)
            execution_instance.stdout_callbacks.add(self._stdout_handle)
            execution_instance.finish()

        log.info("Reset everything at the end")
        self._reset()

    def _etcd_get_keys(self, etcd_key_root, etcd_keys):
        log.info("Parsing '{}' ETCD key values".format(etcd_key_root))
        etcd_values = {}
        for etcd_key in etcd_keys:
            try:
                etcd_values[etcd_key] = \
                self._etcd_get(etcd_key_root, etcd_key)
            except Exception as error:
                log.exception(error)
                raise PipelineError(error)

        return etcd_values
            
    def _etcd_get(self, etcd_key_root, etcd_key):
        try:
            value = self._etcd.get(path.join(self._etcd_root,
                                             "{}/{}".format(etcd_key_root,
                                                            etcd_key)))[0].decode('UTF-8')
            return value
        except Exception as error:
            log.exception(error)
            raise PipelineError(error)    

    def _reset(self):
        commands = ["ipcrm -a",
                    "pkill -9 -f corrsim",
                    "pkill -9 -f udpdb",
                    "pkill -9 -f process",
                    "pkill -9 -f dada_db",
                    "pkill -9 -f dada_diskdb",
                    "pkill -9 -f dada_dbdisk",
                    #"xbutil reset"
        ]
        execution_instances = []
        for command in commands:
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
                log.error(returncode)
                
                log.error("Terminate all execution instances "
                          "when error happens")
                self._terminate_executions()
                
                log.error("Reset when error happens")
                self._reset()
                
                raise PipelineError(returncode)

    def _stdout_handle(self, stdout, callback):
        if self._execution:
            log.debug(stdout)

    def _terminate_executions(self):
        for execution_instance in self._execution_instances:
            execution_instance.terminate()
            
if __name__ == "__main__":
    # Run with ./pipeline.py -a localhost:2379 -b /SB123/beam01
    logging.basicConfig(filename='craco_pipeline.log')
    log = logging.getLogger('craco_pipeline')
    coloredlogs.install(
        fmt=("[ %(levelname)s\t- %(asctime)s - %(name)s "
             "- %(filename)s:%(lineno)s] %(message)s"),
        level='DEBUG',
        #level='INFO',
        #level='WARNING',
        #level='ERROR',
        #level='CRITICAL',
        logger=log)
    
    parser = argparse.ArgumentParser(
        description='To run the CRACO pipeline')
    parser.add_argument('-a', '--etcd_server', type=str, nargs='+',
                        help='ETCD server to configure the pipeline')
    parser.add_argument('-b', '--etcd_root', type=str, nargs='+',
                        help='ETCD root to configure the pipeline')
    
    # Parse top-level pipeline configuration
    log.info("Started to parse "
             "top-level pipeline configuration ...")
    args = parser.parse_args()
    etcd_server = args.etcd_server[0]
    etcd_root   = args.etcd_root[0]
    log.info("ETCD server is {}".format(etcd_server))
    log.info("ETCD root is {}".format(etcd_root))
    log.info("Finished the parsing "
             "of top-level pipeline configuration.")

    log.info("Started to execute the pipeline ...")
    execution = True
    pipeline  = Pipeline(execution,
                         etcd_server,
                         etcd_root)
