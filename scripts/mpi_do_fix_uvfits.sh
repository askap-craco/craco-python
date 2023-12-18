#!/bin/bash
# Fun fixuvfits - needs to be called from
# mpi_run_beam

if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi
indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    exit 1
fi

source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate craco

cmd="fixuvfits $uvfits"
echo running $cmd
$cmd



