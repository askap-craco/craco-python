#!/bin/bash
echo "Hello!"
if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi
indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    # don't exit with nonzero oetehrwise the whole mpi job goes down
    exit 0
fi

rootdir=$(echo $indir | sed s%/data/craco/%/CRACO/DATA_00/%)

source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate calib

uvfits_average $uvfits $@
