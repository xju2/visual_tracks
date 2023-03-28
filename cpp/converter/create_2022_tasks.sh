#!/bin/bash

FILES=/global/homes/x/xju/m3443/data/ITk-upgrade/ReDumpped2022Data/GNN4Itk__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3595_s3600_r12401/GNN4Itk__mc15_14TeV.600012.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.recon.RDO.e8185_s3595_s3600_r12401__J*.root


EXE=/global/cfs/cdirs/m3443/usr/xju/code/visual_tracks/cpp/converter/process_one_file.sh


outname="tasks_2022data.txt"
for FNAME in $FILES
do
	echo $EXE $FNAME >> $outname
done
