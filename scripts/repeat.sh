#!/usr/bin/bash

for i in {0..100}
do
    echo "Loop number: $i start"
    #mpirun -c 2 run_cluster_messages.py --nrx 1 --nlink 1 --method rdma   --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 1_000_000
    mpirun -c 3 run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 1_000_000
    echo "Loop number: $i done"
done
