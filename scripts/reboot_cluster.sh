#!/bin/bash
sudo -v
for f in `cat $HOSTFILE` ; do
    echo rebooting $f
    ssh $f sudo reboot &
done

wait
