#!/bin/bash

rankfile=mpipipeline.rank

if [[ -z $HOSTFILE ]] ; then
    echo "HOSTFILE environment variable not set. Please point toa path for an mpi host file"
    exit 1
fi  
hostfile=$HOSTFILE


echo "Running $0 with argumnets $@ `pwd` with rankfile=$rankfile hostfile=$hostfile"
# IF HCOLL IS ENABED WITH col-hcoll-enable 1 THEN IT CAN HANGS ON MPI_FINALIZE !!!!
# hcoll uses mlx5 even if its not enabled in UCX_TLS

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


# save the rankfile
extra_args="--hostfile $hostfile"

echo "Resetting cards in the background see CRACO-327. It takes about 25 seconds if its all working well. Hopefully that's enough time"
time mpirun -map-by ppr:1:node -hostfile $HOSTFILE `which resetallcards.sh` &> resetlogs.log &

rm rx.rank beam.rank
mpipipeline --dump-rankfile $rankfile $extra_args $@

retval=$?
if [ $retval -ne 0 ]; then
    echo "MPIPipeline returned $retval"
    exit $?
fi

if [[ -z $SCAN_DIR ]] ; then
    SCAN_DIR='.'
fi
echo "Using SCAN_DIR $SCAN_DIR"

export NUMBA_CACHE_DIR=/data/craco/craco/tmpdir/numba_cache
mkdir -p $NUMBA_CACHE_DIR
export NUMBA_DEBUG_CACHE=1


# can add --report-bindings to be verbose
commonargs="-x EPICS_CA_ADDR_LIST -x EPICS_CA_AUTO_ADDR_LIST -x PYTHONPATH -x XILINX_XRT -wdir $SCAN_DIR -x NUMBA_CACHE_DIR -x NUMBA_DEBUG_CACHE"

# runwith the rankfile

echo "UCX_NET_DEVICES=$UCX_NET_DEVICES UCX_TLS=$UCX_TLS"

echo "Making directories"
# directories on host
mpirun -hostfile $hostfile -map-by ppr:1:node mkdir -p $SCAN_DIR
# local directory
mkdir -p $SCAN_DIR


# TODO: MPI can abort explosively if you like by doing `which python` -m mpi4py before `which pipeline`
# but I hve trouble with pyton versions 
pipeline="`which python` -m mpi4py `which mpipipeline` --mpi $extra_args $@"

# its 5 apps now
#    RX - receives data from cards -  per card
#    BEAMPROC - processes beam data - 1 per beam
#    PLANNER - creates plan and sends async to beamproc - 1 per beam
#    BEAM_CAND - recieves candidates from BEAMPROC - 1 per beam
#    CAND_MGR - consolidate candidates from BEAM_CAND  - 1 per application
# at the moment we just run the pipeline 5 times with the same arguments. The ranks files choose where the put them.
mgrhost=skadi-00

# I can't get rankfiles to work. It complaines its oversubsribed or such
# nonsense.
# unfortuantely,I don't think I can do binding very well
#cmd="mpirun $commonargs $tcpargs 
#     -rf rx.rank $pipeline :  
#    -rf beam.rank $pipeline :  
#    -rf beam.rank $pipeline  : 
#    -rf beam.rank $pipeline   :
#    -host $mgrhost -np 1 $pipeline "

ncards=$(cat rx.rank | wc -l)
nbeams=$(cat beam.rank | wc -l)
nhosts=$(cat $HOSTFILE | wc -l)
#pipeline=printenv.sh
echo Pipeline is $pipeline
echo NBEAMS=$nbeams NCARDS=$ncards NHOSTS=$nhosts

#cmd="mpirun --oversubscribe --display-map $commonargs -hostfile $HOSTFILE 
#    $ucxargs -np $ncards -map-by ppr:4:socket -- $pipeline --proc-type rx    :  
#    $ucxargs -np $nbeams -map-by ppr:1:socket -- $pipeline --proc-type beam  :
#    $ucxargs -np $nbeams -map-by ppr:1:socket -- $pipeline --proc-type plan  :
#    $ucxargs -np 1       -map-by ppr:1:socket -- $pipeline --proc-type mgr   : 
#    $ucxargs -np $nbeams -map-by ppr:1:socket -- $pipeline --proc-type cand "

#cmd="mpirun --display-map $commonargs $tcpargs -hostfile $HOSTFILE -map-by ppr:1:socket
#    -np $ncards -- $pipeline --proc-type rx    :  
#    -np $nbeams -- $pipeline --proc-type beam  "

cmd="mpirun  $commonargs -rankfile mpipipeline.rank $ucxargs $pipeline"
echo on `date` running $cmd
$cmd




