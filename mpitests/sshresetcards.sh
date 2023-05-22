#!/bin/bash

for i in {1..10} ; do
    hostname=$(printf "seren-%02d" $i)
    echo $hostname
    ssh $hostname `realpath resetbothcards.sh`
done
