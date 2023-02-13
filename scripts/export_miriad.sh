#!/bin/bash

mir=$1
#mfcal vis=$mir
for opt in gain bandpass ; do
    for yaxis in real imag amp phase; do
        gpplt vis=$mir log=${mir}.${opt}.${yaxis} yaxis=${yaxis} options=${opt} nxy=6,6 device=${mir}.${opt}.${yaxis}.png/png
    done
done
