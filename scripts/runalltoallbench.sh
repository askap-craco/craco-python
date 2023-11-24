#!/bin/bash

which osu_alltoall
#mpirun -v -map-by ppr:8:node -mca pml ucx  -x UCX_TLS=$UCX_TLS -hostfile mpi_seren.txt   `which osu_alltoall`  -m 2:128
#mpirun -v -mca pml ucx  -x UCX_TLS=$UCX_TLS -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_IB_GID_INDEX=0  -host seren-01,seren-02,seren-03,seren-04,seren-05   `which osu_alltoall`  -m 2:2000000


# THis definitly works - the trick is to set UCX_IB_GID_INDEX=0  to use ROCEv1 otherwise everything breaks.
# With UCX_TLS=self,tcp,rc,rc_mlx5,ud,ud_mlx5,mm,cma
# on 25 May 2022 after a site shutdown and reboot
#runign with RoceV1 and not specificying UCX_NET_DEVICES means it rusn about 30 % faster because presumabley it uses both NICS
# WHich, if you look at the ethtoo  -S counters, it  clearly is. This is a good thing

# OK - 2 March 2023 - still driving myself CRAZY.
# Thisactually makes the thing work with ROCE but, of course, as the switch isn't lossless, it's terrible
#ifaces=mlx5_0:1
#enable_hcoll=0
#verbose=0
#export UCX_NET_DEVICES=$ifaces
#export UCX_TLS=self,tcp,mm,cma,rc,rc_mlx5,ud,ud_mlx5
#export UCX_IB_GID_INDEX=3
# Other things might also work
# iF you want to to go realy slowly, don't add teh rc stuff and ud stuff
# ALSO:
# If using TCP you have to use UCX_NET_DEVICES=enp216s0
# AND USING ROCE you have to have their IB name: mlx5_0:1
# Sheesh

use_roce=1
enable_hcoll=1
verbose=0

if [[ $use_roce == 1 ]] ; then
    echo "Setting up for RoCE"
    export UCX_NET_DEVICES=mlx5_2:1,mlx5_0:1
    export UCX_TLS=self,mm,cma,rc,rc_mlx5,ud,ud_mlx5
    export UCX_IB_GID_INDEX=0
else
    echo "Setting up for TCP"
    export UCX_NET_DEVICES=ens3f0np0,ens6f0np0
    export UCX_TLS=self,tcp,mm,cma
fi

echo UCX_TLS=$UCX_TLS
echo UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX
echo UCX_IB_SL=$UCX_IB_SL
echo UCX_NET_DEVICES=$UCX_NET_DEVICES
echo enable_hcoll=$enable_hcoll


commonargs="--mca oob_tcp_if_include eno8303 --mca oob_base_verbose $verbose --mca coll_hcoll_enable $enable_hcoll"
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca pml_ucx_verbose $verbose"
tcpargs="--mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include $ifaces"

hostfile=mpi_skadi.txt
cmd="mpirun -v --map-by ppr:1:socket  $ucxargs $commonargs  -hostfile $hostfile  `which osu_alltoall`  -m 2048:8000000  -f"
#cmd="mpirun -v -map-by ppr:1:node  $ucxargs $commonargs  -host seren-01,seren-02,seren-03  `which osu_alltoall`  -m 2:20000 -f"
echo $cmd
$cmd

