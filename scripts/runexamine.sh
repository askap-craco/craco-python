#!/bin/bash

dout=xbmgmt_logs
mkdir -p $dout
for hostnum in {1..10} ; do
    for device in 0000:86:00.0 0000:3b:00.0 ; do
	host=$(printf 'seren-%02d' "$hostnum")
	fout="$dout/${host}_${device}_xbmgmt.log"
	echo "XBMGMT Results for $host $device" > $fout
	ssh $host sudo `which xbmgmt` examine -d $device | tee -a $fout
    done
done
