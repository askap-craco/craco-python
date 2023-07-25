#!/bin/bash

trap 'kill $(jobs -p)' EXIT
nhost=$1
rate=$2
starthost=$3
if [[ -z $starthost ]] ; then
    starthost=2
fi
endhost=$(($starthost + $nhost - 1))
msgsize=300000
gididx=2
dev=mlx5_1
duration=10
echo $nhost $rate $starthost-$endhost

for i in `seq $starthost $endhost` ; do
    port=$(($i + 18515))
    hostid=$(($i))
    host=seren-0${hostid}
    #host=seren-02
    cmd="ssh $host taskset -c 10-20 ib_send_bw -F -d $dev -p $port --report_gbits -c UC -s $msgsize -x $gididx  10.25.11.31 --rate_limit $rate --rate_units=g -D $duration --rate_limit_type=SW -S $UCX_IB_SL"
    sleep 0.05
    echo $cmd
    $cmd &
done
echo Got jobs $(jobs -p)

wait


	   
