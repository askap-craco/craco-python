#!/bin/bash

trap 'kill $(jobs -p)' EXIT

msgsize=30000000
for i in {1..10} ; do
    port=$(($i + 18515))
    taskset -c 20-30 ib_send_bw -F -d mlx5_0 -p $port --report_gbits -D 1 -c UC -s $msgsize -x 0 --run_infinitely -D 1 &
done

wait


	   
