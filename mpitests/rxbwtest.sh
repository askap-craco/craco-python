#!/bin/bash

trap 'kill $(jobs -p)' EXIT

msgsize=30000000

gididx=0
for i in {1..10} ; do
    port=$(($i + 18515))
    cmd="taskset -c 20-30 ib_send_bw -F -d mlx5_0 -p $port --report_gbits -D 1 -c UC -s $msgsize -x $gididx --run_infinitely -D 1 -S $UCX_IB_SL"
    echo $cmd
    $cmd &
done

wait


	   
