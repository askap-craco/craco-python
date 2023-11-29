#!/bin/bash
# Fun fixuvfits - needs to be called from
# mpi_run_beam

if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi
indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
xrtcardno=$OMPI_COMM_WORLD_LOCAL_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    exit 1
fi

outdir=$(basename $uvfits)/results/

# Andy environment breaks XRT wit this eror
# ImportError: /CRACO/SOFTWARE/craco/wan342/Software/conda3/envs/craco38/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/xilinx/xrt/lib/libxrt_coreutil.so.2)
#    import pyxrt
#ImportError: /CRACO/SOFTWARE/craco/wan342/Software/conda3/envs/craco38/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/xilinx/xrt/lib/libxrt_coreutil.so.2)

cmd="search_pipeline --uv $uvfits --outdir $outdir --device $xrtcardno $@"
echo `hostname` running $cmd
$cmd
