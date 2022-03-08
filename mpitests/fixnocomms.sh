#!/bin/bash

mpirun -np 20 -hostfile mpi_seren.txt pingall.sh
