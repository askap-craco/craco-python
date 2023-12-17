#!/bin/bash
# Runs search pipeline on UVFITS calculated from OMPI_COMM_WORLD_RANK
# and sets device from OMPI_COM_WORLD_LOCAL_RANK
# All other argumetns passed through to search pipeline

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

# fixuvfits $uvfits

### activate my environment
source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate craco

# Andy environment breaks XRT wit this eror
# ImportError: /CRACO/SOFTWARE/craco/wan342/Software/conda3/envs/craco38/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/xilinx/xrt/lib/libxrt_coreutil.so.2)
#    import pyxrt
#ImportError: /CRACO/SOFTWARE/craco/wan342/Software/conda3/envs/craco38/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/xilinx/xrt/lib/libxrt_coreutil.so.2)

cmd="search_pipeline --uv $uvfits --device $xrtcardno $@"
echo `hostname` running $cmd
$cmd
