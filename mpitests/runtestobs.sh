#!/bin/bash

ccap="./mpicardcap.sh   -e --block 2 --prefix ak -N 100"

CARDCAP_DIR=$1

if [[ ! -e $CARDCAP_DIR ]] ; then
    mkdir -p $CARDCAP_DIR
fi

nmsg=100
target=target
cards="1-12 1"
dotscrunch="on off"

for spi in 16 32 64 ; do
    for pol in ps dp ; do
	for beam in 0 -1; do
	    for fpga in 1-6 ; do
		for cards in $cards ; do
		    for ts in  off on; do 
			if [[ $pol == "ps" ]] ; then
			    polcmd="--pol-sum"
			else
			    polcmd="--dual-pol"
			fi
			
			if [[ $ts == "on" ]] ; then
			    tscrunch=$(( 2048 / $spi ))
			else
			    tscrunch="1"
			fi
			
			if [[ $pol == "dp" && $spi == 16 ]] ; then # this mode doesnt work
			    continue 
			fi
			      
			dout="$CARDCAP_DIR/cap_spi${spi}_bm${beam}_a${cards}_k${fpga}_$pol_ts$ts"
			cmd="$ccap --tscrunch $tscrunch --samples-per-integration $spi $polcmd --beam $beam -a $cards -k $fpga --num-msgs $nmsg -f $dout/$target.fits"
			echo $cmd
			$cmd
		    done
		done
	    done
	done
    done
done
	   

