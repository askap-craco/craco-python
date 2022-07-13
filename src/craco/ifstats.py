#!/usr/bin/env python
"""
Prints network stats

Copyright (C) CSIRO 2020
"""
import os
import sys
import logging
import subprocess
import time

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

class NetInterface:
    def __init__(self, ifname):
        self.ifname = ifname

    def get_statistics(self):
        cmd = ['/sbin/ethtool','-S',self.ifname]
        cmd =  ' '.join(cmd)
        p = subprocess.run(cmd, shell=True, capture_output=True, encoding='utf=8', check=True)
        d = {}
        for line in p.stdout.split('\n')[1:]:
            if line.strip() == '':
                continue
            
            try:
                bits = line.split(':')
                d[bits[0].strip()] = int(bits[1])
            except:
                print('Could not parse', line)

        return d

    def poll_stats(self, values):
        d = self.get_statistics()
        counts = ('rx_discards_phy','tx_pause_ctrl_phy','tx_global_pause','rx_global_pause', 'rx_prio3_pause','rx_prio3_pause_duration','tx_prio3_pause','tx_prio3_pause_duration')
        print(self.ifname, values.field, '(Gbps) ', ' '.join(counts))

        while True:
            time.sleep(values.sleep)
            dnext = self.get_statistics()
            diff = dnext[values.field] - d[values.field]
            diffpersec = diff / values.sleep
            scale = 8/1e9
            dout = diffpersec * scale
            countvalues = [ddiff(d, dnext, cf) for cf in counts]
            print(f'{dout:0.1f}', ' '.join(map(str,countvalues)))
            d = dnext

                            
            
    

def ddiff(d1, d2, f):
    try:
        d = d2[f] - d1[f]
    except KeyError:
        d = 'X'

    return d
            
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-s','--sleep', type=float, help='sleep time', default=1.0)
    parser.add_argument('-f','--field', help='Field to print', default='rx_bytes_phy')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    netif =NetInterface(values.files[0])
    netif.poll_stats(values)
    

if __name__ == '__main__':
    _main()
