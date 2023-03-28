#!/bin/bash

FILES=~/m3443/data/ITk-upgrade/BugFixedNewTtbarSamples/v1/*root

EXE=/global/cfs/cdirs/m3443/usr/xju/code/visual_tracks/cpp/converter/process_one_file.sh


outname="tasks_2023data.txt"
for FNAME in $FILES
do
	echo $EXE $FNAME >> $outname
done
