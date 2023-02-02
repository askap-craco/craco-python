#!/bin/bash

f=$1
invert vis=$f map=$f.imap beam=$f.ibeam imsize=2,2,beam cell=5,5,res robust=2 options=sdb,double,mfs

mfclean map=$f.imap beam=$f.ibeam out=$f.icln niters=100 region=perc\(66\) log=$f.iclnlog

restor model=$f.icln beam=$f.ibeam map=$f.imap out=$f.irest
