#!/bin/bash
# ULP1
indir=/data/big/craco/SB049721/scans/00/20230427155734/
caldir=/data/seren-01/big/craco/calibration/SB049719/scans/00/20230427094131/
meta=/data/seren-01/big/craco/SB049721/SB49721.json.gz
fcm=/home/ban115/20220714.fcm

outdir=$indir/results5

#dead=seren-04:0-1,seren-07:0-1,seren-09:1
dead=seren-01:0-1,seren-04:0-1
search_beams=0-19
#search_beams=20-35



./mpipipeline.sh --cardcap-dir $indir \
		 --outdir $outdir \
		 --calibration $caldir \
		 --phase-center-filterbank pc.fil \
		 --flag-ants 13,14,15,24-30 \
		 --metadata $meta \
		 --xclbin $XCLBIN \
		 --pol-sum \
		 --block 2-4 --card 1-12  --max-ncards 30 \
		 --ncards-per-host 3 \
		 --nd 100 -N 10000000 \
		 --search-beams $search_beams \
		 --dead-cards $dead \
		 --threshold 6 \
		 --fcm $fcm \ 
		 2>&1 | tee pipeline.run
		 
