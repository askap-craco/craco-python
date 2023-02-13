#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.utils import ibc2beamchan,beamchan2ibc

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

'''
Shape = (9,4,2)
'''
READOUT_CLK = np.array((
    ((0,7),(1,6),(2,4),(4,6)),
    ((0,6),(1,5),(2,3),(4,5)),
    ((0,5),(1,4),(2,2),(4,4)),
    ((0,4),(1,3),(3,7),(5,7)),
    ((0,3),(1,2),(3,6),(5,6)),
    ((0,2),(1,1),(3,5),(5,5)),
    ((0,1),(2,7),(3,4),(6,7)),
    ((0,0),(2,6),(3,3),(6,6)),
    ((1,7),(2,5),(4,7),(7,7))
    ), dtype=np.int32)


def pol2polidx(p):
    if p == 0 or p == 'X' or p == 'A':
        pout = 0
    elif p == 1 or p == 'Y' or p == 'B':
        pout = 1
    else:
        raise ValueError(f'Invalid polidx {p}')

    assert pout == 0 or pout == 1

    return pout


def polidx2pol(p, poltype=None):
    '''
    Converts pol index in 0,1 to a different one

    >>>polidx2pol(0)
    1

    '''
    if poltype is None:
        pout = p
    elif poltype == 'XY':
        pout = 'XY'[p]
    elif poltype == 'AB':
        pout = 'AB'[p]
    else:
        raise ValueError(f'Invalid poltype {p} {poltype}')

    return pout

class CorrelatorCells:
    def __init__(self):
        self.nbeam = 36
        self.nant = 36
        self.npercell = 9
        self.npolin = 2
        self.npolout = 4
        self.ncells = self.nbeam*self.npolin // self.npercell
        assert self.nbeam*self.npolin == self.ncells*self.npercell

        self.cell_to_ants = {}
        self.ants_to_cell = {}
        self.blidx = {}
        ibl = 0
        for a1 in range(self.nant):
            for a2 in range(a1, self.nant):
                self.blidx[(a1,a2)] = ibl
                ibl += 1

        for a1 in range(self.nant):
            for a2 in range(self.nant):
                for p1 in range(self.npolin):
                    for p2 in range(self.npolin):
                        xidx = p1 + self.npolin*a1
                        yidx = p2 + self.npolin*a2
                        if xidx > yidx: #only do lower diagonal
                            continue
                        xcell = xidx // self.npercell
                        xcellidx = xidx % self.npercell
                        ycell = yidx // self.npercell
                        ycellidx = yidx % self.npercell

                        key = (a1,p1,a2,p2)
                        value = (xcell, xcellidx, ycell, ycellidx)
                        self.ants_to_cell[key] = value
                        self.cell_to_ants[value] = key

    def prodidx(self, antidx, polidx):
        return polidx + self.npolin*antidx

    def get_cell(self, a1, p1, a2, p2, antonebased=False):
        if antonebased:
            a1 -= 1
            a2 -= 1

        assert 0 <= a1 < self.nant
        assert 0 <= a2 < self.nant


        p1 = pol2polidx(p1)
        p2 = pol2polidx(p2)
        xidx = self.prodidx(a1,p1)
        yidx = self.prodidx(a2,p2)
        if xidx > yidx: # swap
            a1, a2 = a2, a1
            p1, p2 = p2, p1

        key = (a1,p1, a2,p2)
        cell = self.ants_to_cell[key]
        return cell

    def get_ants(self, xcell, xcellidx, ycell, ycellidx, antonebased=False, poltype=None):
        assert 0 <= xcell < self.ncells
        assert 0 <= ycell < self.ncells
        assert 0 <= xcellidx < self.npercell
        assert 0 <= ycellidx < self.npercell

        if xcell > ycell:
            raise ValueError(f'This Cell doesnt exist {xcell}{ycell}')

        if xcell == ycell: # on diagonal cells, we should swap if necessary
            if xcellidx > ycellidx:
                # swaap
                tmp = xcellidx
                xcellidx = ycellidx
                ycellidx = tmp

        
        key = (xcell, xcellidx, ycell, ycellidx)

        a1,p1,a2,p2 = self.cell_to_ants[key]
        if antonebased:
            a1 += 1
            a2 += 1

        p1 = polidx2pol(p1, poltype)
        p2 = polidx2pol(p2, poltype)

        r = (a1,p1,a2,p2)
        
        return r
        

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
