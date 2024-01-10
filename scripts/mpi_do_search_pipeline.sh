#!/bin/bash
# Runs search pipeline on UVFITS calculated from OMPI_COMM_WORLD_RANK
# and sets device from OMPI_COM_WORLD_LOCAL_RANK
# All other argumetns passed through to search pipeline

if [[ -z $INDIR ]] ; then
    echo "$0: No input directory"
    exit 1
fi

if [[ -z $RUNNAME ]] ; then
    RUNNAME="results"
    echo "$0: No run name using $RUNNAME"
fi

indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
xrtcardno=$(($START_CARD + $OMPI_COMM_WORLD_LOCAL_RANK))

echo "Running $0 with start_card $START_CARD xrtcard=$xrtcardno"

uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    # don't exit with nonzero oetehrwise the whole mpi job goes down
    exit 0
fi

### activate my environment
source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate craco

fixuvfits $uvfits

cmd="search_pipeline --uv $uvfits --device $xrtcardno $@"
echo `hostname` running $cmd
mkdir $indir/$RUNNAME
logfile=$(printf "$indir/$RUNNAME/search_pipeline_b%02d.log" $beamno)

$cmd 2>&1 | tee $logfile

#VG: Adding the following lines to run candpipe automatically after a search pipeline run has finished
#


# Write data to wrunname
cd $indir/$RUNNAME

# beamno=$(printf "%02d" $beamno)
candfile=$(printf "candidates.b%02d.txt" $beamno)
# echo $candfile, $PWD

cmd="`which candpipe` $candfile --save-rfi -s -o clustering_output -v"
$cmd 
