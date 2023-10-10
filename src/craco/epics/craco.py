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
import time

class Craco(EpicsSubsystem):
    """
    The external CRACO API to TOS (ASKAP IOCs)
    """

    def configure(
        self,
        fpgaMask: int,
        flushOnBeam: bool,
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
        flushOnBeam: bool,
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


    def enable_card(self, block:int, card: int, enable: bool):
        assert 1 <= block <= 7
        assert 1 <= card <= 12
        self.write_correlator_card(block, card, 'F_craco:enablePacketiser_O', enable)
        self.write_correlator_card(block, card, 'F_craco:enableSubsystem_O', enable)

    def enable_card_events(self, block:int, cardlist):
        '''
        Enable CRACO go event for cards in the card list
        cards numbered 1-12 inclusive
        call anytiem before cracoStart. It will persist but not accross IOC restarts
        without a config change to theIOC

        Can use this to disable particular cards from sending data
        '''

        mask = 0
        for c in cardlist:
            assert 1 <= c <= 12
            mask |= 1 << (c - 1)

        self.write(f'acx:s{block:02d}:evtf:WF2:enable_fo1.MSKV', hex(mask), wait=True)

    def enable_events_for_blocks_cards(self, blocklist, cardlist, maxncard=None):
        ''''
        Enable craco go events for given blocks
        cards number 1-12 inclsive
        If a card isn't in the list, it is disabled
        If the block ins't in the list it is disabled
        If the card number numbered from 0 is < maxncard and maxncard is specified, the it only enables the lower cards
        '''
        if maxncard is None:
            maxncard = len(blocklist)*len(cardlist)

        icard = 0
        block_masks = []
        for block in range(2,7+1):
            mask = 0 # everyghing is disabled to start with
            for card in range(1, 12+1):
                if block in blocklist and card in cardlist and icard < maxncard:
                    mask |= 1 << (card - 1)
                    icard += 1

            block_masks.append(mask)

                
        for block, mask in zip(range(2,7+1), block_masks):
            self.write(f'acx:s{block:02d}:evtf:WF2:enable_fo1.MSKV', hex(mask), wait=True)


    def start_block(self, block:int):
        self.write(f'acx:s{block:02d}:evtf:craco:enable', 0, wait=True)
        time.sleep(1)
        self.write(f'acx:s{block:02d}:evtf:craco:enable', 1, wait=True)
        time.sleep(1)
        self.write(f'acx:s{block:02d}:evtf:craco:enable', 0, wait=True)

    def start_async(self, blocklist, cardlist):
        '''
        Hacky way of starting a shelf
        cardlist is the list of cards to shart
        ak:acx:s07:c01:F_craco:enablePacketiser_O
        ak:acx:s07:c01:F_craco:enableSubsystem_O
        ak:acx:s07:evtf:craco:enable

        The correct way to run everything is to call start()
        '''
        for block in range(2,7):
            self.write(f'acx:s{block:02d}:evtf:craco:enable', 0, wait=True)

        for card in range(1, 12+1):
            enable = 1 if card in cardlist else 0
            self.write_correlator_card(block, card, 'F_craco:enablePacketiser_O', enable)
            self.write_correlator_card(block, card, 'F_craco:enableSubsystem_O', enable)

        for block in range(2,7):
            enable = 1 if block in blocklist else 0
            self.write(f'acx:s{block:02d}:evtf:craco:enable', enable, wait=False)

        time.sleep(200)
        for block in range(2,7):
            self.write(f'acx:s{block:02d}:evtf:craco:enable', 0, wait=False)

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

    def read_start_bat(self):
        '''
        Retuns the value of start bat as an int
        '''
        batstr = self.read('evtf:craco:startBat', cache=False)
        if batstr is None or len(batstr) == 0:
            bat = 0
        else:
            bat = int(batstr, 16)
        return bat

    def get_start_bat(self):
        if self.start_bat is not None:
            return self.start_bat
        
        for retry in range(100):
            bat = self.read_start_bat()
            if bat != self.previous_start_bat:
                self.start_bat = bat
                break
            
            time.sleep(0.1)

        if self.start_bat is None:
            raise ValueError('timeout getting start bat')

        return self.start_bat

    def start(self):
        """
        start CRACO
        """
        en = self.read('enableCraco', cache=False)
        if en != 1:
            raise ValueError('CRACO not enabled in EPICS. Ask someone to enable it via CSS or add it to an SB parset and schedule another block. Sorry kids. No dice')

        zoomval = self.read('S_zoom:val', cache=False)
        standard_zoom = zoomval == 0
        if not standard_zoom:
            raise ValueError(f'Correlator is not in standard zoom. Its in zoom mode={zoomval}. Please wait until operations schedules an SB in standard zoom')

        
        self.previous_start_bat = self.read_start_bat()
        self.start_bat = None
        self.write("cracoStart", 1, timeout=10.0, wait=False)

    def stop(self):
        """
        stop CRACO
        """
        self.write("cracoStop", 1, timeout=10.0)

    def get_channel_frequencies(self, block: int, card: int):
        return self.read_correlator_card(block, card, 'F_processCorrelations:skyFreqList')
