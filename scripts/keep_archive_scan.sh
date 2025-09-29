#!/bin/bash
sbid=$1
scanid=$2
tstart=$3

comment="acacia archiving"
echo $comment >> /CRACO/DATA_00/craco/$sbid/KEEP

### run fixuvfits...
echo "adding writing access to uvfits file for fixuvfits"
cmd="chmod a+w /CRACO/DATA_??/craco/$sbid/scans/$scanid/$tstart/*.uvfits"
echo "executing $cmd"
$cmd

for uvpath in /CRACO/DATA_??/craco/$sbid/scans/$scanid/$tstart/*.uvfits
do
    echo "running fixuvfits on $uvpath"
    `which fixuvfits` $uvpath
done

###
cmd="chmod a-w /CRACO/DATA_??/craco/$sbid/scans/$scanid/$tstart/*.uvfits*"
echo "executing $cmd"
$cmd