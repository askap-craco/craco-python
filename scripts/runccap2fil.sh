#!/bin/bash

# arg 1 looks like this /data/big/craco/SB046820/scans/00
# scanid looks l
target=/data/seren-10/big/craco/results
thedir=$1

if [[ ! -d $thedir ]] ; then
    echo Argument $thedir is not a direcotry
    exit 1
fi

# scanid looks like SB046776/scans/16/20221220071128
scanid=$(echo $thedir | sed s%/data/big/craco/%%)
outdir=$target/$scanid
echo $1 $thedir $outdir
mkdir -p $outdir
cd $outdir

if [[ ! -d files ]] ; then
    mkdir files
    cmd="ln -s /data/seren-*/big/craco/$scanid/*.fits files"
    echo Linking files with $cmd
    $cmd
else
    echo "Files already exist"
fi

if [[ -e ics.fil ]]  ; then
    echo ics.fil already exists
else
    ccapfits2fil files/*.fits
fi

