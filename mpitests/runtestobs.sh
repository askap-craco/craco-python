#!/bin/bash

ccap="./mpicardcap.sh -e --block 7 --prefix ak"

export CARDCAP_DIR=/data/fast/ban115/testdata/nosb/

for spi in 16 32 64 ; do
    for pol in ps dp ; do
	if [[ $pol == "ps" ]] ; then
	    polcmd="--pol-sum"
	else
	    polcmd="--dual-pol"
	fi
	for beam in 0 -1 ; do
	    for fpga in 1 1-6 ; do
		for cards in 1 1-12 ; do
		    $ccap --samples-per-integration $spi $polcmd --beam $beam -a $cards -k $fpga --num-msgs 100 -f $CARDCAP_DIR/cap_spi${spi}_bm${beam}_a${cards}_k${fpga}.fits
		done
	    done
	done
    done
done
	   

