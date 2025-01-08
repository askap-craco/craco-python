#!/bin/bash

# Dumb but works
for f in $@ ; do
    id=$(realpath $f | awk -F/ '{print $5"_"$7"_"$8}')
    curl --cacert ~/.config/craco_elastic_ca.crt -H 'Content-Type: application/json' $CRAFT_ELASTIC_URL/cracoscans/_create/${id} -X POST -d @${f}
done