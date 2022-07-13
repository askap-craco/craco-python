#!/bin/bash

mpirun --map-by ppr:2:node -hostfile mpi_seren.txt `which osu_alltoall`
