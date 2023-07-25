#!/bin/bash

for line in `env | grep MPI_ | grep RANK` ; do
    echo `hostname` PID $$ $@ $line
done

for line in `env | grep EPICS` ; do
    echo `hostname` PID $$ $@ $line
done

for line in `env | grep UCX` ; do
    echo `hostname` PID $$ $@ $line
done


