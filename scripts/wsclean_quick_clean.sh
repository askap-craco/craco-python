#/bin/bash

### load stuff from command line

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ms)
            if [[ -z "$2" ]]; then
                echo "Error: -ms requires a Measurement Set argument."
                exit 1
            fi
            ms="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$ms" ]]; then
    echo "Please use wsclean_quick_clean.sh -ms <Measurement Set> to specify the Measurement Set to clean."
    exit 1
fi

if [[ "$ms" != *.ms ]]; then
    echo "Error: '$ms' is not a Measurement Set (.ms)"
    exit 1
fi

echo "The script will run wsclean on $ms"

# activate the correct environment
echo "activating conda environment for wsclean..."
source /home/craftop/.conda/.remove_conda.sh
source /home/craftop/.conda/.activate_conda.sh
conda activate wsclean

workdir=$(dirname "$ms")
basename=$(basename "$ms" .ms)

echo "working directory: $workdir"
echo "measurement set basename: $basename"

### making directory for output
imagedir="$workdir/image"
mkdir -p $imagedir
echo "wsclean output directory: $imagedir"

imageparam="-size 5120 5120 -scale 2.5asec"
selectparam=" -minuvw-m 100 -pol xx"
cleanparam="-niter 100 -auto-threshold 2.0 -auto-mask 3.0 -weight briggs -0.5"
otherparam="-use-wgridder -parallel-reordering 8"
# -parallel-deconvolution 
outputparam="-name $imagedir/$basename"

cmd="`which wsclean` $imageparam $selectparam $cleanparam $otherparam $outputparam $ms"
echo "running $cmd"
$cmd

