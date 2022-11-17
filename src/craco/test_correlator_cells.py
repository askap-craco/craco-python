
from correlator_cells import *

def test_pol2polidx():
    assert pol2polidx(0) == 0
    assert pol2polidx(1) == 1
    assert pol2polidx('A') == 0
    assert pol2polidx('B') == 1
    assert pol2polidx('X') == 0
    assert pol2polidx('Y') == 1

    
def test_correlator_get_cell():
    c = CorrelatorCells()
    assert c.get_cell(0,0,0,0) == (0,0,0,0)
    assert c.get_cell(0,'X',0,'X') == (0,0,0,0)
    assert c.get_cell(1,'X',1,'X', antonebased=True) == (0,0,0,0)
    assert c.get_cell(1,'A',5,'A', antonebased=True) == (0,0,0,8)
    assert c.get_cell(5,'A',1,'A', antonebased=True) == (0,0,0,8)
    assert c.get_cell(5,'B',5,'A', antonebased=True) == (0,8,1,0)
    assert c.get_cell(6,'A',8,'B', antonebased=True) == (1,1,1,6)


def test_correaltor_get_ant():
    c = CorrelatorCells()
    assert c.get_ants(0,0,0,0, antonebased=False, poltype=None) == (0,0,0,0)
    assert c.get_ants(0,0,0,0, antonebased=True, poltype=None) == (1,0,1,0)
    assert c.get_ants(0,0,0,0, antonebased=True, poltype='XY') == (1,'X',1,'X')
    assert c.get_ants(0,0,0,0, antonebased=True, poltype='AB') == (1,'A',1,'A')
    assert c.get_ants(0,0,0,8, antonebased=True, poltype='AB') == (1,'A',5,'A')
    assert c.get_ants(0,8,0,0, antonebased=True, poltype='AB') == (1,'A',5,'A') # transposed
    assert c.get_ants(0,8,1,0, antonebased=True, poltype='AB') == (5,'A',5,'B')
    


def test_ibc2beamchan():
    assert ibc2beamchan(0) == (0,0)
    assert ibc2beamchan(1) == (1,0)
    assert ibc2beamchan(2) == (2,0)
    assert ibc2beamchan(31) == (31,0)
    assert ibc2beamchan(32) == (0,1)
    assert ibc2beamchan(33) == (1,1)
    assert ibc2beamchan(32*2) == (0,2)
    assert ibc2beamchan(32*3) == (0,3)
    assert ibc2beamchan(32*4) == (32,0)
    assert ibc2beamchan(32*4 + 1) == (33,0)
    assert ibc2beamchan(36*4 - 1) == (35,3)

    
    


def test_readout_clk():
    print(READOUT_CLK.shape)
    assert READOUT_CLK.shape == (9,4,2)
    s = set()
    for i in range(READOUT_CLK.shape[0]):
        for j in range(READOUT_CLK.shape[1]):
            x,y = READOUT_CLK[i,j,:]
            assert 0 <= x < 8
            assert 0 <= y < 8
            s.add((x,y))

    assert len(s) == 9*4 # check XY is unique
    

    

    
