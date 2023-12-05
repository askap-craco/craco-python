#!/bin/bash
#rankfile=host1_fpga72_rankfile.txt
#rankfile=host1_fpga6_rankfile.txt
#rankfile=host12_fpga6_rankfile.txt
#rankfile=host11_fpga6_rankfile.txt # seren-10 disk is dead
#rankfile=host1_fpga12_rankfile.txt
#rankfile=host1_fpga36_rankfile.txt
#rankfile=$1
#shift

rankfile=mpicardcap.rank

if [[ -z $HOSTFILE ]] ; then
    echo "HOSTFILE environment variable not set. Please point toa path for an mpi host file"
    exit 1
fi
hostfile=$HOSTFILE

use_roce=0
enable_hcoll=0
verbose=0

if [[ $use_roce == 1 ]] ; then
    echo "Setting up for RoCE"
    export UCX_NET_DEVICES=mlx5_2:1,mlx5_0:1
    export UCX_TLS=self,mm,cma,rc,rc_mlx5,ud,ud_mlx5
    export UCX_IB_GID_INDEX=0
else
    echo "Setting up for TCP"
    if [[ $(hostname) == "athena" ]] ; then
        dev=eno1
    fi

    export UCX_NET_DEVICES=ens3f0np0,ens6f0np0
    export UCX_TLS=self,tcp,mm,cma
fi

# save the rankfile
cardcap --hostfile $hostfile --dump-rankfile $rankfile $@

echo "Created rankfile $rankfile"

commonargs="--mca oob_tcp_if_include eno8303 --mca oob_base_verbose $verbose --mca coll_hcoll_enable $enable_hcoll"
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca pml_ucx_verbose $verbose"
tcpargs="--mca pml ob1 --mca btl tcp,self" # --mca btl_tcp_if_include $ifaces"

# runwith the rankfile
# add --report-bindings
cmd="mpirun -rf $rankfile   -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST $commonargs $ucxargs `which cardcap` --mpi $@"
echo $cmd
$cmd

echo $0 $cmd is FINISHED
exit 0




