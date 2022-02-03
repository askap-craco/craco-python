#!/bin/bash

export WORKDIR=/data/seren-01/fast/den15c
# run with mpi_seren.sh
#source $WORKDIR/venv3.7/bin/activate; mpirun -c 3 $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# run with mpirun -c 3 mpi_seren.sh
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# run with mpirun -c 2 mpi_seren.sh
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 1 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 2 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 2 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 10_000
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 10 --nlink 10 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 1 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 10_000
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 1 --method mpi --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 10 --nlink 10 --method mpi --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 10_000


#mpirun -map-by ppr:2:node --rank-by node -hostfile /data/seren-01/fast/den15c/craco-python/mpitests/mpi_seren.txt /data/seren-01/fast/den15c/craco-python/mpitests/mpi_seren.sh

# Good command as follow:

# one transmitter and one receiver running on the same node, use all 10 nodes
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 10 --nlink 10 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# one transmitter and one receiver running on the same node, use 2 nodes
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 2 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# one transmitter and one receiver running on different nodes, use 2 nodes
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 1 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# one transmitter and one receiver running on different nodes, use 4 nodes
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 2 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000

# two transmitters and two receivers running on the same nodes, use 1 node
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 1_000_000

# two transmitters and two receivers running on the same nodes, use 2 nodes
source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 2 --nlink 4 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 1_000_000

# Two transmitters and two receivers running on seperate nodes, receivers run on the same node, transmitters run on seperate nodes, with one pair of three
#source $WORKDIR/venv3.7/bin/activate; $WORKDIR/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65_536 --num-blks 10 --num-cmsgs 100 --nmsg 1_000_000 #--send-delay 1_000_000
