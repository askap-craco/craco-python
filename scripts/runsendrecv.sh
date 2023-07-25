#!/bin/bash
#SBATCH --job-name=all2all
#SBATCH --time=00:01:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=2g

# Application specific commands:
module load openmpi
module load python/3.7.2
source venv/bin/activate

mpirun -bind-to none ./sendrecv.py 

