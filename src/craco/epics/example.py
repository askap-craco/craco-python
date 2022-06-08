#!/usr/bin/env python
"""
CRACO example using simple caproto based class
"""
import time

from craco import Craco

PREFIX = "ma:" # "ch:" I think is for ... testing?
CORRELATOR_BLOCKS = [1]
NUM_CARDS = 1
NUM_FPGAS = 6

# get the Craco interface
craco = Craco(PREFIX)

# set the ROCE headers
for block in CORRELATOR_BLOCKS:
    for card in range(1, 1 + NUM_CARDS):
        for fpga in range(1, 1 + NUM_FPGAS):
            craco.set_roce_header(block, card, fpga, [42] * 612)

# configure CRACO on all FPGAS
craco.configure(0x3F, True, True, True, 31, 1, 1)

# start CRACO (enabling packetiser, craco subsystem and firing event)
craco.start()

# observe
time.sleep(10)

# stop CRACO (disabled packetiser &  craco subsystem)
craco.stop()
