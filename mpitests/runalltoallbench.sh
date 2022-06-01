#!/bin/bash

which osu_alltoall
#mpirun -v -map-by ppr:8:node -mca pml ucx  -x UCX_TLS=$UCX_TLS -hostfile mpi_seren.txt   `which osu_alltoall`  -m 2:128
#mpirun -v -mca pml ucx  -x UCX_TLS=$UCX_TLS -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_IB_GID_INDEX=0  -host seren-01,seren-02,seren-03,seren-04,seren-05   `which osu_alltoall`  -m 2:2000000


# THis definitly works - the trick is to set UCX_IB_GID_INDEX=0  to use ROCEv1 otherwise everything breaks.
# With UCX_TLS=self,tcp,rc,rc_mlx5,ud,ud_mlx5,mm,cma
# on 25 May 2022 after a site shutdown and reboot
#runign with RoceV1 and not specificying UCX_NET_DEVICES means it rusn about 30 % faster because presumabley it uses both NICS
# WHich, if you look at the ethtoo  -S counters, it  clearly is. This is a good thing

echo UCX_TLS=$UCX_TLS
mpirun -v -map-by ppr:1:node  -mca pml ucx   -x UCX_TLS=$UCX_TLS  -x UCX_IB_GID_INDEX=0  -hostfile mpi_seren.txt  `which osu_alltoall`  -m 2:262144 -f

