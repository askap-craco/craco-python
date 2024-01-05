#!/bin/bash
# Fun fixuvfits - needs to be called from
# mpi_run_beam
echo "Hello!"
if [[ -z $INDIR ]] ; then
    echo "No input directory"
    exit 1
fi
indir=$INDIR
beamno=$OMPI_COMM_WORLD_RANK
uvfits=$(printf "$indir/b%02d.uvfits" $beamno)
if [[ ! -f $uvfits ]] ; then
    echo "UVFITS not found! $uvfits"
    exit 1
fi

nsamp=5000
outfits=$uvfits.uvw.uvfits
rootdir=$(echo $indir | sed s%/data/craco/%/CRACO/DATA_00/%)
metafile=$rootdir/../../../SB*.json.gz

echo "Got fits $uvfits writing to $outfits with metafile $metafile $rootdir"

source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate calib

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/CRACO/SOFTWARE/craco/wan342/Software/conda3/envs/craco38/lib
# export PATH=$PATH:/CRACO/SOFTWARE/craco/wan342/bin
echo $LD_LIBRARY_PATH

logfile=${uvfits}.cal.log

fixuvfits $uvfits 

if [[ ! -f $outfits ]] ; then
    #attach_uvws_uvfits -outname $outfits -end_samp $nsamp $uvfits $metafile
    quicksnip -O $outfits -N $nsamp $uvfits --metadata $metafile
fi


cmd="/CRACO/SOFTWARE/craco/wan342/Software/craco_calib/calib_skadi.py -uv $outfits"
echo running $cmd
$cmd 



