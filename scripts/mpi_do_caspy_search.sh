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

ics=$(printf "$indir/ics_b%02d.fil" $beamno)
if [[ ! -f $ics ]] ; then
    echo "ICS not found! $ics"
    exit 1
fi
candfile="${ics}.cand"
cmd="search_cas_fil -f $ics -C $candfile"
echo running $cmd
$cmd
