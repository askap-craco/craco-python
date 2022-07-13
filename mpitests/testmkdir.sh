#!/bin/bash

fname=/data/big/testdata/SB41783/spi64-dp-ball-ts32-v2/tst.txt
touch  $fname
if [[ $? != 0 ]] ; then
    echo `hostname` could not touch $fname
fi
