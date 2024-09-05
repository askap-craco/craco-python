#!/bin/bash

# redirect everything to a log file
# also use XBALL and don't run in parallel - it might confuse thigns.
source /opt/xilinx/xrt/setup.sh
nrank=$OMPI_COMM_WORLD_NODE_RANK

echo "Starting $0. NRANK=$nrank arg1=$1"
    
dev="all"
    
#redict to log file and stdout
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

echo `hostname` `date` showing xball examine
xball xbutil examine 
 
echo `hostname` `date` running reset
xball xbutil reset --force

echo `hostname` `date` Reset returned $?

xball xbutil examine

echo `hostname` `date` $dev=$nrank reset complete. 

