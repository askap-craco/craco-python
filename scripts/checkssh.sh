#!/bin/bash

# ssh to every machine and print the hostname
# if it isnt printed then the host is down as far as SSH is concertn

trap 'killall `jobs -p` ; exit' INT

for f in `cat $HOSTFILE` ; do
    ssh -o ConnectTimeout=10 $f hostname &
done

wait
