#!/bin/bash

source /data/fast/den15c/venv3.7/bin/activate; mpirun -c 3 /data/fast/den15c/craco-python/mpitests/run_cluster_messages.py --nrx 1 --nlink 2 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000
