#!/bin/bash

which osu_alltoall
#mpirun -v -map-by ppr:8:node -mca pml ucx  -x UCX_TLS=$UCX_TLS -hostfile mpi_seren.txt   `which osu_alltoall`  -m 2:128
#mpirun -v -mca pml ucx  -x UCX_TLS=$UCX_TLS -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_IB_GID_INDEX=0  -host seren-01,seren-02,seren-03,seren-04,seren-05   `which osu_alltoall`  -m 2:2000000


# THis definitly works - the trick is to set UCX_IB_GID_INDEX=0  to use ROCEv1 otherwise everything breaks.
echo UCX_TLS=$UCX_TLS
mpirun -v -map-by ppr:1:node  -mca pml ucx  -x UCX_TLS=$UCX_TLS  -x UCX_IB_GID_INDEX=0  -hostfile mpi_seren.txt  `which osu_alltoall`  -m 2:2000

