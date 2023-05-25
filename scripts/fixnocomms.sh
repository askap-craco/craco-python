#!/bin/bash

mpirun -map-by ppr:1:node -hostfile mpi_seren.txt pingall.sh
