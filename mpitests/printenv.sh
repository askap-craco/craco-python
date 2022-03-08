#!/bin/bash

for line in `env | grep MPI_ | grep RANK` ; do
    echo `hostname` PID $$ $@ $line
done
