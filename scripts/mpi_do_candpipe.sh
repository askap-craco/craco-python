#!/bin/bash
# run cand pipeline 

if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi

indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
# provide runname as the additional parameter

source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate craco

# note this assume we are using results folder!
cd $indir/results
# beamno=$(printf "%02d" $beamno)
candfile=$(printf "candidates.b%02d.txt" $beamno)
# echo $candfile, $PWD

cmd="`which candpipe` $candfile --save-rfi -s -o clustering_output -v"
$cmd
