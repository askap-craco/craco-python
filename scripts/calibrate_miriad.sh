#!/bin/bash

myfits=$1
vis=$myfits.mir

fits in=$myfits out=$myfits.mir op=uvin
uvflag vis=$vis select=amp\(0,0.0000001\) flagval=flag
mfcal vis=$vis
export_miriad.sh $vis
