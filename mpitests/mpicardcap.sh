#!/bin/bash
#rankfile=host1_fpga72_rankfile.txt
#rankfile=host1_fpga6_rankfile.txt
#rankfile=host12_fpga6_rankfile.txt
#rankfile=host11_fpga6_rankfile.txt # seren-10 disk is dead
#rankfile=host1_fpga12_rankfile.txt
#rankfile=host1_fpga36_rankfile.txt
rankfile=$1
shift

rankfile=mpicardcap.rank
hostfile=mpi_seren.txt

# save the rankfile
cardcap --hostfile $hostfile --dump-rankfile $rankfile $@
ifaces=enp216s0,enp175s0

enable_hcoll=0
verbose=0
commonargs="--mca oob_tcp_if_include eno1 --mca oob_base_verbose $verbose --mca coll_hcoll_enable $enable_hcoll"
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca pml_ucx_verbose $verbose"
tcpargs="--mca pml ob1 --mca btl tcp,vader,self" # --mca btl_tcp_if_include $ifaces"

# runwith the rankfile
cmd="mpirun -rf $rankfile   --report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST $commonargs $ucxargs `which cardcap` --mpi $@"
echo $cmd
$cmd




