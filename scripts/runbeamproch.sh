#!/bin/bash



mpirun --mca btl_base_verbose 30 \
       --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_IB_GID_INDEX=3 \
       -hostfile mpi_seren.txt -np 164 /data/seren-01/fast/ban115/build/venv/bin/python ./beamproc.py --nrx 144 --nbeam 20 --map-by ppn:10
