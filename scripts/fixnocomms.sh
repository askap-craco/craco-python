#!/bin/bash


mpirun -map-by ppr:1:node -hostfile ~/mpihosts.txt `which pingall.sh`
