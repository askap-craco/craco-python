#!/bin/bash

mpirun --hostfile mpi_seren.txt --map-by ppr:2:node -x PATH -x XILINX_XRT resetcards.sh

