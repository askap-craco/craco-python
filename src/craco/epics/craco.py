#!/usr/bin/env python
from .pyepics import EpicsSubsystem


class Craco(EpicsSubsystem):
    """
    The external CRACO API to TOS (ASKAP IOCs)
    """

    def configure(
        self,
        fpgaMask: int,
        enMultiDest: bool,
        enPktzrDbgHdr: bool,
        enPktzrTestData: bool,
        lsbPosition: int,
        sumPols: int,
        integSelect: int,
    ):
        """
        Configure CRACO
        """
        self.call_ioc_function("cracoConfigure", locals())

    def set_roce_header(self, block: int, card: int, fpga: int, headers: list):
        """
        Write the CRACO ROCE Headers

        :param block: block number 2..7
        :param card: card number 1..12
        :param fpga: fpga number 1..6
        :param headers: headers in beam order 36 * 17 words
        """
        return self.write_correlator_card(
            block, card, f"F_craco:fpga{fpga}:writeRoceHeaders_O", headers
        )

    def start(self):
        """
        start CRACO
        """
        self.write("cracoStart", 1, timeout=10.0)

    def stop(self):
        """
        stop CRACO
        """
        self.write("cracoStop", 1)

    def get_channel_frequencies(self, block: int, card: int):
        return self.read_correlator_card(block, card, 'F_processCorrelations:skyFreqList')
