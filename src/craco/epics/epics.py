#!/usr/bin/env python
from typing import Any

import caproto.sync.client


class EpicsSubsystem:
    """
    Simple Wrapper for EPICS PVs using Caproto synchronous client

    """

    def __init__(self, prefix: str):
        self._prefix = prefix
        # self._ctx = Context()

    def read(self, pvname: str):
        """
        read from an EPICS PV
        """
        return caproto.sync.client.read(f"{self._prefix}{pvname}")

    def write(self, pvname: str, value: Any, timeout: float = 5.0):
        """
        write to EPICS PV
        """
        print(f"{self._prefix}{pvname} = {value}")
        return caproto.sync.client.write(
            f"{self._prefix}{pvname}", value, notify=True, timeout=timeout
        )

    def write_correlator_card(self, block: int, card: int, pvname: str, value: Any):
        """
        write to a correlator card PV
        """
        return self.write(f"acx:s{block:02d}:c{card:02d}:{pvname}", value)

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
