#!/bin/bash

# redirect everything to a log file
source /opt/xilinx/xrt/setup.sh
nrank=$OMPI_COMM_WORLD_NODE_RANK

echo "Starting $0. NRANK=$nrank arg1=$1"
    
if [[ $nrank  == 0 ]] ; then
    dev=0000:86:00.1
elif [[ $nrank == 1 ]] ; then
    dev=0000:3b:00.1
else
    dev=$1
    echo "Set device rom arg1 $dev"
fi

if [[ ! -z $dev ]] ; then
    echo "Device $dev not set"
fi
    
log_file="resetlogs/$(hostname)_${dev}.log"
mkdir -p $(dirname $log_file)
exec &> >(tee  "$log_file")

echo "Resetting card $dev nrank=$nrank"

xbmgmt=/opt/xilinx/xrt/bin/xbmgmt
image=xilinx_u280_gen3x16_xdma_base_1

bdf=$(echo $dev | sed s/.1/.0/)

echo `hostname` `date` attempting reset $dev=$nrank

echo `hostname` `date` Showing examine
xbutil examine

xbutil examine -d $dev
 
xbutil reset -d $dev --force

echo `hostname` `date` Reset returned $?

xbutil examine -d $dev

echo `hostname` `date` $dev=$nrank reset complete. 

