#!/usr/bin/env python
from typing import Any

from epics import caput, caget


class EpicsSubsystem:
    """
    Simple Wrapper for EPICS PVs using Caproto synchronous client

    """

    def __init__(self, prefix: str, pvcache: dict = {}):
        self._prefix = prefix
        self.cache = pvcache
        assert self.cache is not None
        # self._ctx = Context()

    def read(self, pvname: str, timeout:float = 5.0, cache=True):
        """
        read from an EPICS PV
        """
        key = f"{self._prefix}{pvname}"

        if cache:
            value = self.cache.get(key, None)
        else:
            value = None
            
        if value is None:
            value = caget(key, timeout=timeout)
            if value is None:
                raise ValueError(f'Could not key {key}')
            self.cache[key] = value
            
        return value

    def write(self, pvname: str, value: Any, timeout: float = 5.0, wait=True):
        """
        write to EPICS PV
        """
        print(f"{self._prefix}{pvname} = {value}")
        return caput(
            f"{self._prefix}{pvname}", value, timeout=timeout, wait=wait
        )

    def write_correlator_card(self, block: int, card: int, pvname: str, value: Any):
        """
        write to a correlator card PV
        """
        return self.write(f"acx:s{block:02d}:c{card:02d}:{pvname}", value)

    def read_correlator_card(self, block: int, card: int, pvname: str):
        """
        write to a correlator card PV
        """
        return self.read(f"acx:s{block:02d}:c{card:02d}:{pvname}")

    def call_ioc_function(self, funcname: str, parameters: dict):
        """
        execute an ASKAP style IOC function

        :param funcname:    function name (should match PV name)
        :param parameters:  dictionary of parameters.  The dict
                            keys should match the PV names of
                            the parameters.  If the parameter
                            names of the calling function match
                            the PV names then calling local()
                            will return an appropraite dictionary
        """
        for param, value in parameters.items():
            if "self" == param:
                continue
            self.write(f"F_{funcname}:{param}_O", value)

        self.write(f"F_{funcname}:exec_O", 1)
