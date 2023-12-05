#!/bin/bash

dout=xbtest_logs
mkdir -p $dout
devices="-d 0000:17:00.1 -d 0000:65:00.1 -d 0000:b1:00.1 -d 0000:ca:00.1"
for host in $(cat mpi_skadi.txt) ; do
    fout="$dout/${host}_xbtest.log"
    logout="$(pwd)/$dout/$host"
    mkdir -p $logout
    cmd="ssh $host /opt/xilinx/xbtest/bin/xbtest $devices -l $logout -f -F $@"
    echo "Running $cmd"
    echo "XBTEST Results for $host $device command is $cmd" > $fout
    $cmd | tee -a $fout &
done
wait
