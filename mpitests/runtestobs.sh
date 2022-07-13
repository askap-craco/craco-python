#!/bin/bash

ccap="./mpicardcap.sh -e --block 7 --prefix ak"

CARDCAP_DIR=$1

if [[ ! -e $CARDCAP_DIR ]] ; then
    mkdir -p $CARDCAP_DIR
fi

nmsg=10
target=target
cards=1-12

for spi in 16 32 64 ; do
    for pol in ps dp ; do
	for beam in -1 ; do
	    for fpga in 1 1-6 ; do
		for cards in 1 $cards ; do
		    if [[ $pol == "ps" ]] ; then
			polcmd="--pol-sum"
		    else
			polcmd="--dual-pol"
		    fi

		    if [[ $pol == "dp" && $spi == 16 ]] ; then # this mode doesnt work
			continue 
		    fi

		    
		    $ccap --samples-per-integration $spi $polcmd --beam $beam -a $cards -k $fpga --num-msgs $nmsg -f $CARDCAP_DIR/cap_spi${spi}_bm${beam}_a${cards}_k${fpga}_$pol/$target.fits
		done
	    done
	done
    done
done
	   

