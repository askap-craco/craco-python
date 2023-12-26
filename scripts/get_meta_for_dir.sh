#!/bin/bash

for indir in $@ ; do
    sbid=$(echo $indir | cut -d / -f 5)
    short_sbid=$(echo $indir | sed s/SB0//)
    pushd $indir
    getmeta.sh $short_sbid
    popd
done
