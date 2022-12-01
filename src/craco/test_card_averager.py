#!/usr/bin/env python
from craco.card_averager import Averager
from craco.cardcap import CardcapFile
from craco.cardcapmerger import CcapMerger
import glob

def test_averaging():
    cardfiles = glob.glob('/data/craco/ban115/craco-python/notebooks/data/SB43128/run3/1934_b07_c01+f?.fits')
    assert len(cardfiles) == 6

    cfiles = [CardcapFile(f) for f in cardfiles]
    merger = CcapMerger(cardfiles)
    fid, blk = next(merger.block_iter())

    fileblocks = [next(f.packet_iter()) for f in cfiles]
    from IPython import embed
    embed()
    
    

    
    
