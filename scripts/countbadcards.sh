#!/bin/bash

# counts number of good cards in each block from a 
# run with countbadcareds.sh path/to/run.log

for f in $@ ; do
    echo $f
    grep '0->' $f | cut -d ' ' -f 7 | cut -d / -f 1 | sort | uniq -c

done
