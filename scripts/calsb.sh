#!/bin/bash

for sb in $@ ; do
    prepsb $sb
    pushd $CRACO_RESULTS/$sb
    if [[ ! -e *.json.gz ]] ; then
	getmeta.sh ${sb:3}
    fi

    for s in `ls -d scans/*/*` ; do
	pushd $s
	
	mkdir -p cal
	pushd cal
	job1=$(tsp ccapfits2uvfits -f ~/*.fcm -m ../../../../*.json.gz --beam -2 ../nodes/*/*.fits)
	job2=$(tsp -D $job1 calibrate_miriad.sh *.uvfits)
	popd
	
	popd
    done
    popd
done
