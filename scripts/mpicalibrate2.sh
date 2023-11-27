#!/bin/bash

nbeam=36

# /data/craco/ban115/craco/SB054940/scans/00/20231124034717/mpihosts.txt
indir=$1

if [[ ! -d  $indir ]] ; then
    echo "Input directory doesn't exist"
    exit 1
fi

hostfile=$indir/mpihosts.txt
rankfile=$indir/mpipipeline.rank

if [[ ! -f $hostfile ]] ; then
    echo "Hostfile $hostfile doesnt exit"
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
mpirun -rankfile $beam_only_rankfile $rootdir/dompicalibrate.sh $indir

