#!/bin/bash

dout=xbmgmt_logs
mkdir -p $dout
for device in 0000:17:00.0 0000:65:00.0 0000:b1:00.0 0000:ca:00.0 ; do
    for host in $(cat mpi_skadi.txt) ; do
	fout="$dout/${host}_${device}_xbmgmt.log"
	cmd="ssh $host sudo /opt/xilinx/xrt/bin/xbmgmt -d $device $@"
	echo "Running $cmd"
	echo "XBMGMT Results for $host $device command is $cmd" > $fout
	$cmd | tee -a $fout &
    done
    wait
done
