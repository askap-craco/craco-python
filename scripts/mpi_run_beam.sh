#!/bin/bash
indir=$1
shift  # don't pass indir to mpirun

if [[ ! -d  $indir ]] ; then
    echo "Input directory doesn't exist"
    echo "$0: INDIR CMD [ARGS...]"
    echo Runs an mpi job that launches the 1 process per beam on the correct node
    echo "uses the rankfile from the job to work out the beam <-> host mapping"
    echo Arguments;
    echo indir = local directory containing mpipieline.rank on skadi-00 and
    echo The data on all the other nodes e.g. /data/craco/craco/SB054986/scans/00/20231127051745/
    echo the remaining arguments are passed to mpirun as-is. e.g. the script you want to run and. #
    echo the script is also passed INDIR is set as an environment variable with -x

    exit 1
fi

hostfile=$indir/mpihosts.txt
rankfile=$indir/mpipipeline.rank

if [[ ! -f $hostfile ]] ; then
    echo "Hostfile $hostfile doesnt exist"
    exit 1
fi

if [[ ! -f $rankfile ]] ; then
    echo "Rankfile doesnt exist $rankfile"
    exit 1
fi

# Nice hack - just create rankfile based on the mpipipeline- handy that beams are rank 0-35
beam_only_rankfile=$indir/beam_only.rank
beam_rank_file $rankfile > $beam_only_rankfile

rootdir=$(dirname $0)

# Setup environment for networking

use_roce=1
enable_hcoll=1
verbose=0
oob_verbose=0
export HCOLL_VERBOSE=$verbose
if [[ $use_roce == 1 ]] ; then
    echo "Setting up for RoCE"
    export UCX_NET_DEVICES=mlx5_0:1,mlx5_2:1
    export UCX_TLS=self,mm,cma,rc,rc_mlx5,ud,ud_mlx5
    export UCX_IB_GID_INDEX=0
else
    echo "Setting up for TCP"
    if [[ $(hostname) == "athena" ]] ; then
        dev=eno1
    fi

    export UCX_NET_DEVICES=ens3f0np0,ens6f0np0
    export UCX_TLS=self,tcp,mm,cma
fi
ifaces=$UCX_NET_DEVICES
# use OB1 and TCP
tcpargs=" --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include $ifaces --mca oob_tcp_if_include $ifaces --mca coll_hcoll_enable $enable_hcoll -x coll_hcoll_np=0 --mca orte_base_help_aggregate 0"

# USE UCX
ucxargs="--mca pml ucx -x UCX_TLS -x UCX_IB_GID_INDEX -x UCX_NET_DEVICES --mca oob_tcp_if_include eno8303 --mca oob_base_verbose $oob_verbose --mca coll_hcoll_enable $enable_hcoll -x HCOLL_VERBOSE --mca pml_ucx_verbose $verbose"


echo UCX_TLS=$UCX_TLS
echo UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX
echo UCX_IB_SL=$UCX_IB_SL
echo UCX_NET_DEVICES=$UCX_NET_DEVICES



cmd="mpirun $ucxargs -wdir $indir -rankfile $beam_only_rankfile -x INDIR=$indir -x START_CARD -x RUNNAME $@ "
echo running $cmd with indir=$indir start_card=$START_CARD
$cmd

