#!/bin/bash

rankfile=mpicalibrate.rank
hostfile=mpi_seren.txt

echo "Running $0 with argumnets $@ `pwd` with rankfile=$rankfile hostfile=$hostfile"

# IF HCOLL IS ENABED WITH col-hcoll-enable 1 THEN IT HANGS ON MPI_FINALIZE !!!!
use_roce=0
enable_hcoll=0
verbose=0

if [[ $use_roce == 1 ]] ; then
    echo "Setting up for RoCE"
    export UCX_NET_DEVICES=mlx5_0:1
    export UCX_TLS=self,tcp,mm,cma,rc,rc_mlx5,ud,ud_mlx5
    export UCX_IB_GID_INDEX=3
else
    echo "Setting up for TCP"
    dev=enp175s0.883
    if [[ $(hostname) == "athena" ]] ; then
        dev=eno1
    fi
    export UCX_NET_DEVICES=$dev
    export UCX_TLS=self,tcp,mm,cma
fi

echo UCX_TLS=$UCX_TLS
echo UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX
echo UCX_IB_SL=$UCX_IB_SL
echo UCX_NET_DEVICES=$UCX_NET_DEVICES


# save the rankfile
extra_args="--devices mlx5_1 --hostfile $hostfile"
extra_args=""

mpicalibrate --dump-rankfile $rankfile $extra_args $@

retval=$?
if [ $retval -ne 0 ]; then
    echo "MPIPipeline returned $retval"p
    exit $?
fi



ifaces=eno1
# use OB1 and TCP
tcpargs=" --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include $ifaces --mca oob_tcp_if_include $ifaces --mca coll_hcoll_enable $enable_hcoll -x coll_hcoll_np=0 --mca orte_base_help_aggregate 0"

# USE UCX
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca oob_tcp_if_include eno1 --mca oob_base_verbose $verbose --mca coll_hcoll_enable $enable_hcoll -x HCOLL_VERBOSE --mca pml_ucx_verbose $verbose"

commonargs="--report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST -x PYTHONPATH -x XILINX_XRT"

# runwith the rankfile

echo "UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS"

# TODO: MPI can abort explosively if you like by doing `which python` -m mpi4py before `which pipeline`
# but I hve trouble with pyton versions 
cmd="mpirun --rankfile $rankfile  $commonargs $tcpargs `which python` -m mpi4py `which mpicalibrate`  $extra_args $@"
echo $cmd
$cmd




