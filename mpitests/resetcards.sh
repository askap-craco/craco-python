#!/bin/bash

# redirect everything to a log file
source /opt/xilinx/xrt/setup.sh
nrank=$OMPI_COMM_WORLD_NODE_RANK
    
#if [[ -z $nrank ]] ; then
#    if [[ $nrank  == 0 ]] ; then#
	dev=0000:86:00.1
#    else#
#	dev=0000:3b:00.1
#    fi
#    echo Set device $dev from OMPI rank $nrank
#else
    dev=$1
#fi
    
log_file="$(dirname $0)/resetlogs/$(hostname)_${dev}.log"

exec &> >(tee  "$log_file")

xbmgmt=/opt/xilinx/xrt/bin/xbmgmt
image=xilinx_u280_gen3x16_xdma_base_1

bdf=$(echo $dev | sed s/.1/.0/)

echo `hostname` `date` attempting reset $dev=$nrank

echo Showing examine
xbutil examine -d $dev

echo Showing platform
sudo $xbmgmt examine --device $bdf --report platform

echo Programming...
sudo $xbmgmt program --device $bdf --base --image $image --force

echo Showing platform again
sudo $xbmgmt examine --device $bdf --report platform

echo Showing examine again
xbutil examine -d $dev

#xbutil reset -d $dev --force
echo `hostname` `date` $dev=$nrank reset complete. Rebooting

