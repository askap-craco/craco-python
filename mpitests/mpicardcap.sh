#!/bin/bash
echo $UCX_TLS
#rankfile=host1_fpga72_rankfile.txt
#rankfile=host1_fpga6_rankfile.txt
rankfile=host12_fpga6_rankfile.txt
#rankfile=host11_fpga6_rankfile.txt # seren-10 disk is dead
#rankfile=host1_fpga12_rankfile.txt
#rankfile=host1_fpga36_rankfile.txt
mpirun -rf $rankfile  --hostfile mpi_seren.txt --report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST -mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX  `which cardcap` --mpi $@
