#!/bin/bash


rankfile=mpipipeline.rank

if [[ -z $HOSTFILE ]] ; then
    echo "HOSTFILE environment variable not set. Please point toa path for an mpi host file"
    exit 1
fi  
hostfile=$HOSTFILE


echo "Running $0 with argumnets $@ `pwd` with rankfile=$rankfile hostfile=$hostfile"
# IF HCOLL IS ENABED WITH col-hcoll-enable 1 THEN IT CAN HANGS ON MPI_FINALIZE !!!!
# hcoll uses mlx5 even if its not enabled in UCX_TLS

use_roce=1
enable_hcoll=1
verbose=0
oob_verbose=0


export HCOLL_VERBOSE=$verbose
if [[ $use_roce == 1 ]] ; then
    echo "Setting up for RoCE"
    export UCX_NET_DEVICES=mlx5_0:1,mlx5_2:1
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
ifaces=$UCX_NET_DEVICES
# use OB1 and TCP
tcpargs=" --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include $ifaces --mca oob_tcp_if_include $ifaces --mca coll_hcoll_enable $enable_hcoll -x coll_hcoll_np=0 --mca orte_base_help_aggregate 0"

# USE UCX
# don't do: --mca oob_tcp_if_include eno8303
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES  --mca oob_base_verbose $oob_verbose --mca coll_hcoll_enable $enable_hcoll -x HCOLL_VERBOSE --mca pml_ucx_verbose $verbose"


echo UCX_TLS=$UCX_TLS
echo UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX
echo UCX_IB_SL=$UCX_IB_SL
echo UCX_NET_DEVICES=$UCX_NET_DEVICES


# save the rankfile
extra_args="--hostfile $hostfile"

cmd="mpirun --map-by ppr:5:socket --rank-by socket:span -hostfile $HOSTFILE $ucxargs --display-map --report-bindings `which python` -m mpi4py /CRACO/SOFTWARE/ban115/craco-python/src/craco/mpi_transposer.py $@"
echo $cmd
$cmd