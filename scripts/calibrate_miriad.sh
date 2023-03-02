#!/bin/bash

myfits=$1
vis=$myfits.mir


echo Calibrating $myfits into $vis in directory `pwd`

if [[ ! -e $myfits.mir ]] ; then
    fits in=$myfits out=$myfits.mir op=uvin
    uvflag vis=$vis select=amp\(0,0.0000001\) flagval=flag
fi
mfcal vis=$vis select=uvrange\(0.3,1000\)
export_miriad.sh $vis
