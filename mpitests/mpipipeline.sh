#!/bin/bash
echo $UCX_TLS

rankfile=mpipipeline.rank
hostfile=mpi_seren.txt

# save the rankfile
mpipipeline --hostfile $hostfile --dump-rankfile $rankfile $@

# runwith the rankfile
mpirun -rf $rankfile   --report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST -mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES `which mpipeline` --mpi $@

#mpirun --map-by ppr:48:node -oversubscribe --hostfile mpi_seren.txt --report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST -mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES `which cardcap` --mpi $@



