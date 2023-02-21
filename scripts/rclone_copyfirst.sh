#!/bin/bash

# copy first few chunks of the given set of files using RCLONE
sbid=$1
shift
count=40

nbl=465
nbeam=36
nchan=4
nbytes=4
ntimes=64
bs=$((nbl * nbeam * nchan * ntimes * nbytes))

size=$((bs * count))
total_size=$((size * $# / 1024 / 1024 / 1024))

echo "Copying $sbid Block size is $bs bytes. Count=$count size=$size total size=$total_size GB"

for f in $@ ; do
    outf=acacia:craco/$sbid/`basename $f`
    echo $outf
    dd if=$f bs=$bs count=40 | rclone rcat --size $size $outf
done

echo "FINISHED copying $sbid Block size is $bs bytes. Count=$count size=$size total size=$total_size GB"

