#!/bin/bash

#mpirun -hostfile mpi_seren.txt -map-by ppr:1:node runmpiccapfits2np.sh $@
thedir=$1

ls $thedir/*.fits | parallel ccapfits2np -w 

