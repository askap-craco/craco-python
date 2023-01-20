#!/bin/bash

# find list of source addresses
local_addresses=$(ifconfig | grep inet | grep 10.25 | awk '{print $2}')

# for each source address
for la in $local_addresses ; do

    # for a bunch of destinations on the same subnet
    for i2 in `seq 31 40` ; do
	# calculate destination address
	destaddr=$(awk -F"." '{print $1"."$2"."$3}' <<< $la)
	destaddr=${destaddr}.$i2
	cmd="ping -I $la -c 1 -i 1  $destaddr"
	$cmd > /dev/null
	sleep 1
    done
done
