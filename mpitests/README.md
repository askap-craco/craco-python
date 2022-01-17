# MPI to test RoCE on SEREN clusterl

Assume that
1. We have a Python virtual environment setup and it is at `/data/seren-01/fast/den15c/venv3.7`;
2. We have a Python script `run_cluster_messages.py` at `/data/seren-01/fast/den15c/craco-python/mpitests` to launch MPI transmitters and receivers with given options.

We have following user cases:
- Case 1: Two transmitters and one receiver running on the same node, but only use one node;
- Case 2: One transmitter and one receiver running on the same node, use all 10 nodes;
- Case 3: One transmitter and one receiver running on seperate node, only use 2 nodes;


## Case 2:

1. Write the hostname of all nodes along with `slots=2` into a file like `mpi_seren.txt`, the file should looks like as follow. 
```
seren-01 slots=2
seren-02 slots=2
seren-03 slots=2
seren-04 slots=2
seren-05 slots=2
seren-06 slots=2
seren-07 slots=2
seren-08 slots=2
seren-09 slots=2
seren-10 slots=2
```

2. Run `source /data/seren-01/fast/den15c/venv3.7/bin/activate` to bring up the Python3 virtual environment;
3. Run '/data/seren-01/fast/den15c/craco-python/mpitests/run_cluster_messages.py --nrx 10 --nlink 10 --method rdma --msg-size 65536 --num-blks 10 --num-cmsgs 100 --nmsg 100_000` to launch 10 transmitters and 10 receivers running on 10 nodes, each node has one transmitter and one receiver talking to each other.

Once the execution is done, we will be able to see print out information as follow:
```
INFO:__main__:Rank 0 transmitter elapsed time is 0.5373530387878418 seconds
INFO:__main__:Rank 0 receiver from transmitter 0, elapsed time is 0.5373380184173584 seconds
INFO:__main__:Rank 0 transmitter data rate is 97.56863033336262 Gbps

INFO:__main__:Rank 0 receiver from transmitter 0, data rate is 97.57135769849394 Gbps
INFO:__main__:Rank 0 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 0 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 0 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 0 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 1 transmitter elapsed time is 0.5383594036102295 seconds
INFO:__main__:Rank 1 transmitter data rate is 97.38624355479502 Gbps

INFO:__main__:Rank 1 receiver from transmitter 0, elapsed time is 0.5383477210998535 seconds
INFO:__main__:Rank 1 receiver from transmitter 0, data rate is 97.38835690227698 Gbps
INFO:__main__:Rank 1 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 1 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 1 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 1 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 3 transmitter elapsed time is 0.5423808097839355 seconds
INFO:__main__:Rank 3 receiver from transmitter 0, elapsed time is 0.5423851013183594 seconds
INFO:__main__:Rank 3 transmitter data rate is 96.66418695913245 Gbps

INFO:__main__:Rank 3 receiver from transmitter 0, data rate is 96.66342211938137 Gbps
INFO:__main__:Rank 3 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 3 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 3 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 3 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 9 transmitter elapsed time is 0.542844295501709 seconds
INFO:__main__:Rank 9 receiver from transmitter 0, elapsed time is 0.5428245067596436 seconds
INFO:__main__:Rank 9 transmitter data rate is 96.58165413996682 Gbps

INFO:__main__:Rank 9 receiver from transmitter 0, data rate is 96.58517503745436 Gbps
INFO:__main__:Rank 9 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 9 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 9 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 9 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 6 transmitter elapsed time is 0.5433382987976074 seconds
INFO:__main__:Rank 6 transmitter data rate is 96.4938420796463 Gbps

INFO:__main__:Rank 6 receiver from transmitter 0, elapsed time is 0.5433351993560791 seconds
INFO:__main__:Rank 6 receiver from transmitter 0, data rate is 96.49439252626142 Gbps
INFO:__main__:Rank 6 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 6 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 6 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 6 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 2 transmitter elapsed time is 0.5436379909515381 seconds
INFO:__main__:Rank 2 transmitter data rate is 96.44064777046404 Gbps

INFO:__main__:Rank 2 receiver from transmitter 0, elapsed time is 0.5436296463012695 seconds
INFO:__main__:Rank 2 receiver from transmitter 0, data rate is 96.44212812291133 Gbps
INFO:__main__:Rank 2 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 2 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 2 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 2 receiver from transmitter 0, message loss rate is 0.0

INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:__main__:Rank 5 transmitter elapsed time is 0.5442266464233398 seconds
INFO:__main__:Rank 5 transmitter data rate is 96.33633403392196 Gbps

INFO:__main__:Rank 5 receiver from transmitter 0, elapsed time is 0.5442161560058594 seconds
INFO:__main__:Rank 5 receiver from transmitter 0, data rate is 96.3381910320125 Gbps
INFO:__main__:Rank 5 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 5 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 5 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 5 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 8 receiver from transmitter 0, elapsed time is 0.5444185733795166 seconds
INFO:__main__:Rank 8 receiver from transmitter 0, data rate is 96.30237204168942 Gbps
INFO:__main__:Rank 8 transmitter elapsed time is 0.5444128513336182 seconds
INFO:__main__:Rank 8 transmitter data rate is 96.30338422681989 Gbps

INFO:__main__:Rank 8 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 8 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 8 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 8 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 4 receiver from transmitter 0, elapsed time is 0.5450468063354492 seconds
INFO:__main__:Rank 4 receiver from transmitter 0, data rate is 96.19137180620903 Gbps
INFO:__main__:Rank 4 transmitter elapsed time is 0.5450639724731445 seconds
INFO:__main__:Rank 4 transmitter data rate is 96.18834237403792 Gbps

INFO:__main__:Rank 4 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 4 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 4 receiver from transmitter 0, message total is 100000
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:__main__:Rank 4 receiver from transmitter 0, message loss rate is 0.0

INFO:__main__:Rank 7 transmitter elapsed time is 0.5458018779754639 seconds
INFO:__main__:Rank 7 receiver from transmitter 0, elapsed time is 0.5457911491394043 seconds
INFO:__main__:Rank 7 transmitter data rate is 96.05829901955174 Gbps

INFO:__main__:Rank 7 receiver from transmitter 0, data rate is 96.0601872761568 Gbps
INFO:__main__:Rank 7 receiver from transmitter 0, message missed is 0
INFO:__main__:Rank 7 receiver from transmitter 0, message received is 100000
INFO:__main__:Rank 7 receiver from transmitter 0, message total is 100000
INFO:__main__:Rank 7 receiver from transmitter 0, message loss rate is 0.0

INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
INFO:	Receive Visibilities ending 1
INFO:	Receive Visibilities ending 0
```

The above print out information tells us that we successfully finish the test and the bandwidth on each node is about 100~Gbps.

Please be aware that, for the test here:
1. The number of nodes in `mpi_seren.txt` should matach the number given by `--nrx` and `--nlink`;
2. Make sure that all nodes in `mpi_seren.txt` are up and running fine;
3. The test here only for throughput check, not for test with result comparison, like the `--test=ones` or `--test=increment`. Test with result comparison will harm the performance so that receiver will not be able to receive all packets. Which will cause the script hangs as receivers may wait for missed packets forever. 