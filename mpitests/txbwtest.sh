#!/bin/bash

trap 'kill $(jobs -p)' EXIT
nhost=$1
rate=$2
starthost=$3
if [[ -z $starthost ]] ; then
    starthost=2
fi
endhost=$(($starthost + $nhost - 1))
msgsize=30000000
gididx=2
echo $nhost $rate $starthost-$endhost
dev=mlx5_0
for i in `seq $starthost $endhost` ; do
    port=$(($i + 18515))
    hostid=$(($i))
    host=seren-0${hostid}
    #host=seren-02
    cmd="ssh $host taskset -c 30-40 ib_send_bw -F -d $dev -p $port --report_gbits -c UC -s $msgsize -x $gididx  10.25.11.31 --rate_limit $rate -D 10 --rate_limit_type=SW -S $UCX_IB_SL"
    echo $cmd
    $cmd &
done
echo Got jobs $(jobs -p)

wait


	   
