#!/bin/bash
resetcards=`dirname $0`/resetcards.sh
echo resetting both cards on `hostname` with $resetcards

$resetcards 0000:86:00.1
$resetcards 0000:3b:00.1

sudo reboot
