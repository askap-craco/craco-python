#!/bin/bash

trap 'kill $(jobs -p)' EXIT
nhost=$1
rate=$2
msgsize=30000000
for i in `seq 1 $nhost` ; do
    port=$(($i + 18515))
    hostid=$(($i + 1))
    host=seren-0${hostid}
    #host=seren-02
    cmd="ssh $host taskset -c 30-40 ib_send_bw -F -d mlx5_0 -p $port --report_gbits -c UC -s $msgsize -x 0  10.25.11.31 --rate_limit $rate -D 10"
    echo $cmd
    $cmd &
done
echo Got jobs $(jobs -p)

wait


	   
