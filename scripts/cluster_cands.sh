#!/bin/bash

sbid=$1
sourcedirs=`ls -1d /data/seren-??/big/craco/$1/scans/*/*/results`

for sourcedir in $sourcedirs; do
  echo $sourcedir
  destdir=$sourcedir/clustering_output
  mkdir -p $destdir
  cd $destdir

  for num in `seq -f %02g 0 35`; do
    inpfile=$sourcedir/candidates.txtb$num
    if [[ -f "$inpfile" ]]; then
      python /home/gup037/Codes/craco_clustering/data_handler.py -p $inpfile -m 1 -l "total_sample,dm,lpix,mpix"
      mv unq_cands.dbscan candidates.txtb$num.uniq
      mv unq_cands_dbscan.all candidates.txtb$num.all
    fi
  done

done
