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
		for card in $cards ; do
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

			if [[ $beam == -1 ]] ; then
			    beamcmd=""
			else
			    beamcmd="--beam $beam"
			fi
			      
			dout="$CARDCAP_DIR/cap_spi${spi}_bm${beam}_a${card}_k${fpga}_${pol}_ts$ts"
			cmd="$ccap --tscrunch $tscrunch --samples-per-integration $spi $polcmd $beamcmd -a $card -k $fpga --num-msgs $nmsg -f $dout/$target.fits"
			if [[ -d $dout ]] ; then
			    echo "Directory $dout already exists. Skipping"
			    continue
			fi
			mkdir -p $dout
			echo $cmd
			
			echo "`date` running $cmd" > $dout/run.log
			#timeout -k 1m 1m $cmd 2>&1 | tee -a $dout/run.log
			echo "`date` completed $cmd with return code $? pipe status ${PIPESTATUS} p0=${PIPESTATUS[0]} p1=${PIPESTATUS[1]}" >> $dout/run.log
		    done
		done
	    done
	done
    done
done
	   

