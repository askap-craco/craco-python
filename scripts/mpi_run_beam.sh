#!/bin/bash
indir=$1
shift  # don't pass indir to mpirun

if [[ ! -d  $indir ]] ; then
    echo "Input directory doesn't exist"
    echo "$0: INDIR CMD [ARGS...]"
    echo Runs an mpi job that launches the 1 process per beam on the correct node
    echo "uses the rankfile from the job to work out the beam <-> host mapping"
    echo Arguments;
    echo indir = local directory containing mpipieline.rank on skadi-00 and
    echo The data on all the other nodes e.g. /data/craco/craco/SB054986/scans/00/20231127051745/
    echo the remaining arguments are passed to mpirun as-is. e.g. the script you want to run and. #
    echo the script is also passed INDIR is set as an environment variable with -x

    exit 1
fi

hostfile=$indir/mpihosts.txt
rankfile=$indir/mpipipeline.rank

if [[ ! -f $hostfile ]] ; then
    echo "Hostfile $hostfile doesnt exist"
    exit 1
fi

if [[ ! -f $rankfile ]] ; then
    echo "Rankfile doesnt exist $rankfile"
    exit 1
fi

# Nice hack - just create rankfile based on the mpipipeline- handy that beams are rank 0-35
beam_only_rankfile=$indir/beam_only.rank
grep Beam $rankfile > $beam_only_rankfile

rootdir=$(dirname $0)
cmd="mpirun -rankfile $beam_only_rankfile -x INDIR=$indir -x START_CARD $@ "
echo running $cmd with indir=$INDIR start_card=$START_CARD
$cmd

