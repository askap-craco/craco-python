#!/usr/bin/env python
import sys
import logging
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import numpy as np
import timeit
import time
from netifaces import interfaces, ifaddresses, AF_INET
from subprocess import check_call

def get_address(ifaceName):
    addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':None}] )]
    return addresses[0]


def _main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    print(f'My rank is {rank}/{numprocs}')
    for ifaceName in interfaces():
        addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )]
        print(ifaceName, ' '.join(addresses))

    address  = get_address('ib1')
    print(f'rank={rank} ib1 address {address}')

    address = comm.bcast(address)

    print(f'rank={rank} tx address ={address}')
    
    if rank == 0:
        cmd = f'ib_send_bw -i 2'
    else:
        cmd = f'ib_send_bw -i 2 {address} -D 10 '

    if rank != 0:
        time.sleep(1)

    check_call(cmd, shell=True)

if __name__ == '__main__':
    _main()
