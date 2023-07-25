#!/bin/bash

rankfile=mpipipeline.rank
hostfile=mpi_seren.txt

# save the rankfile
#export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
mpi_transpose_test --hostfile $hostfile --dump-rankfile $rankfile $@
ifaces="enp175s0.88"
ifaces="enp216s0"

# use OB1 and TCP

# IF HCOLL IS ENABED WITH col-hcoll-enable 1 THEN IT HANGS ON MPI_FINALIZE !!!!
enable_hcoll=0

# use UCX and set UX values
export UCX_NET_DEVICES=$ifaces
export UCX_TLS=self,tcp

tcpargs=" --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include $ifaces --mca oob_tcp_if_include $ifaces --mca coll_hcoll_enable $enable_hcoll -x coll_hcoll_np=0 --mca orte_base_help_aggregate 0"

verbose=0
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca oob_tcp_if_include eno1 --mca oob_base_verbose $verbose --mca coll_hcoll_enable $enable_hcoll -x HCOLL_VERBOSE --mca pml_ucx_verbose $verbose"

commonargs="--report-bindings  -x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST"

# runwith the rankfile
cmd="mpirun $commonargs $ucxargs -rf $rankfile `which mpi_transpose_test` --mpi $@"
echo $cmd
$cmd




