#!/usr/bin/env python
# At MATES we need:
# export EPICS_CA_AUTO_ADDR_LIST=no
# export EPICS_CA_ADDR_LIST=202.9.15.255
# with
#from .epics import EpicsSubsystem
# 
# 
# AT MRO We need
# export EPICS_CA_NAME_SERVERS="alderman:35021 bolton:41037
#
# and
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

    def configure_card(
        self,
        card: int,
        fpgaMask: int,
        enMultiDest: bool,
        enPktzrDbgHdr: bool,
        enPktzrTestData: bool,
        lsbPosition: int,
        sumPols: int,
        integSelect: int,
    ):
        """
        Configure CRACO single card
        KB - need to check. Might not work
        """
        d = locals()
        del d['card']
        self.call_ioc_function(f"c:{card:02d}:F_cracoConfigure", locals())


    def start_shelf(self, block:int, cardlist):
        '''
        Hacky way of starting a shelf
        cardlist is the list of cards to shart
        ak:acx:s07:c01:F_craco:enablePacketiser_O
        ak:acx:s07:c01:F_craco:enableSubsystem_O
        ak:acx:s07:evtf:craco:enable

        The correct way to run everything is to call start()
        '''
        self.write(f'acx:s{block:02d}:evtf:craco:enable', 0, wait=True)
        for card in range(1, 12+1):
            enable = 1 if card in cardlist else 0
            self.write_correlator_card(block, card, 'F_craco:enablePacketiser_O', enable)
            self.write_correlator_card(block, card, 'F_craco:enableSubsystem_O', enable)

        self.write(f'acx:s{block:02d}:evtf:craco:enable', 1, wait=False)



    def enable_craco(self, block: int, card: int, fpga:int, enable: bool):
        '''
        ak:acx:s07:cNN:F_cracoConfigure:fpgaMask_O
        '''
        self.write_correlator_card(
            block, card, f"F_cracoConfigure:fpgaMask_O", 0x3f
        )

        

    def set_roce_header(self, block: int, card: int, fpga: int, headers: list):
        """
        Write the CRACO ROCE Headers

        :param block: block number 1..7
        :param card: card number 1..12
        :param fpga: fpga number 1..6
        :param headers: headers in beam order 36 * 17 words
        """
        assert 1 <= block <= 7, f'Invalid block {block}'
        assert 1 <= card <= 12, f'Invalid card {card}'
        assert 1 <= fpga <= 6, f'Invalid FPGA {fpga}'
        return self.write_correlator_card(
            block, card, f"F_craco:fpga{fpga}:writeRoceHeaders_O", headers
        )

    def start(self):
        """
        start CRACO
        """
        self.write("cracoStart", 1, timeout=10.0, wait=False)

    def stop(self):
        """
        stop CRACO
        """
        self.write("cracoStop", 1, timeout=10.0)

    def get_channel_frequencies(self, block: int, card: int):
        return self.read_correlator_card(block, card, 'F_processCorrelations:skyFreqList')
