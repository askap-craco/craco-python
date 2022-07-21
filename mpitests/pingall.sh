#!/bin/bash

for i1 in 10 11 ; do
    for i2 in `seq 31 40` ; do
	ping -c 3 -i 0.5  10.25.$i1.$i2 &
    done
done
wait
