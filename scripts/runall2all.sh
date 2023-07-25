#!/bin/bash
#SBATCH --job-name=all2all
#SBATCH --time=00:01:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=36
#SBATCH --mem=2g

# Application specific commands:
module load openmpi
module load python/3.7.2
#mpirun -bind-to none --mca btl openib,sm,self ./all2all.py # this works at 20Gbps`
mpirun -bind-to none ./all2all.py # this works at 20Gbps`

