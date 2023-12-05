#!/bin/bash

# calibrates a particular beam. lauched by mpicalibrate2.sh
indir=$1
#ls $indir
#printenv.sh
# beam number is the rank
beamno=$OMPI_COMM_WORLD_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    exit 1
fi
echo "Processing on $(hostname) $(ls -lh $uvfits)"
