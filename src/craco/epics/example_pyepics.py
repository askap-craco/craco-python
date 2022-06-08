#!/usr/bin/env python
"""
example CRACO script for ASKAP style py-epics library
"""
import time

from askap.epics.subsystems import AdeCor, Cmp
from askap.event import AbortEvent
from askap.parset import ParameterSet

PREFIX = "ch"
CORRELATOR_BLOCKS = [1]
NUM_CARDS = 1
NUM_FPGAS = 6

CONFIG = ParameterSet(
    {
        "cmp.prefix": f"{PREFIX}:",
        "central.acx1.prefix": f"{PREFIX}:acx:s01:",
        "central.acx2.prefix": f"{PREFIX}:acx:s02:",
        "central.acx3.prefix": f"{PREFIX}:acx:s03:",
        "central.acx4.prefix": f"{PREFIX}:acx:s04:",
        "central.acx5.prefix": f"{PREFIX}:acx:s05:",
        "central.acx6.prefix": f"{PREFIX}:acx:s06:",
        "central.acx7.prefix": f"{PREFIX}:acx:s07:",
    }
)

parset = ParameterSet()
abort = AbortEvent()
composite = Cmp("cmp", CONFIG, parset, abort)
correlator = [AdeCor(f"central.acx{x}", CONFIG, parset) for x in CORRELATOR_BLOCKS]

for block in correlator:
    for card in range(1, 1 + NUM_CARDS):
        for fpga in range(1, 1 + NUM_FPGAS):
            block.craco_set_roce_header(card, fpga, [42] * 612)
composite.craco_configure(0x3F, True, True, True, 31, 1, 42)
composite.craco_start()
time.sleep(10)
composite.craco_stop()
