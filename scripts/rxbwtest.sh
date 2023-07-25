#!/bin/bash

trap 'kill $(jobs -p)' EXIT

msgsize=300000
gididx=2
dev=mlx5_1
for i in {1..3} ; do
    port=$(($i + 18515))
    cmd="taskset -c 10-20 ib_send_bw -F -d $dev -p $port --report_gbits -D 1 -c UC -s $msgsize -x $gididx --run_infinitely -D 1 -S $UCX_IB_SL"
    echo $cmd
    $cmd &
done

wait


	   
