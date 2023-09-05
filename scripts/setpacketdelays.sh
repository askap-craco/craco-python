#!/bin/bash

# https://confluence.csiro.au/display/CRACO/Time+division+multiplexing+CRACO+output
# clock rate is 312.5 MHz= 3.2ns
# 300ns is 94 counts
# 48 us is 
nclk=$1
ncard_per_host=6
for ib in {0..5} ; do
    for ic in {0..11} ; do
	for link in {0..1} ; do
	    # rpoke --site ma -S acx -s 1 -c 2 -m 3 -r -d -a 6 -w 1500
	    b=$(($ib + 2))
	    c=$(($ic + 1))
	    fpga=$(($link + 3))
	    
	    
	    totalcard=$((ib*12 + ic))
	    hostcard=$(($totalcard % $ncard_per_host))
	    ihost=$(($totalcard / $ncard_per_host))
	    
	    nic=$(( $ic % 2 ))
	    
	    icrd=$(( $hostcard / 2 ))

	    tslot=$(( $icrd* 2 + $link))

	    delay=$(($nclk * tslot))

	    
	    cmd="rpoke --site ak -S acx -s $b -m $fpga -c $c -r -d -a 6 -w $delay"
	    echo Block $b card $c totalcard=$totalcard  ihost=$ihost hostcard=$hostcard icrd=$icrd link=$link fpga=$fpga tslot=$tslot  delay=$delay

	    echo $cmd
	    $cmd
	done
    done
done
