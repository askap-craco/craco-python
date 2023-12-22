#!/bin/bash
# Runs search pipeline on UVFITS calculated from OMPI_COMM_WORLD_RANK
# and sets device from OMPI_COM_WORLD_LOCAL_RANK
# All other argumetns passed through to search pipeline

if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi
indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
xrtcardno=$OMPI_COMM_WORLD_LOCAL_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    exit 1
fi

### activate my environment
source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate craco

fixuvfits $uvfits

cmd="search_pipeline --uv $uvfits --device $xrtcardno $@"
echo `hostname` running $cmd
$cmd

#VG: Adding the following lines to run candpipe automatically after a search pipeline run has finished
#


# note this assume we are using results folder!
cd $indir/results

# beamno=$(printf "%02d" $beamno)
candfile=$(printf "candidates.b%02d.txt" $beamno)
# echo $candfile, $PWD

cmd="`which candpipe` $candfile --save-rfi -s -o clustering_output -v"
$cmd
