#!/bin/bash
FNAME=$1
# EXE=/global/cfs/cdirs/m3443/usr/xju/code/converter/bin/ROOT2CSVconverter.exe
# EXE=/global/cfs/cdirs/m3443/usr/xju/code/test/converter/bin/ROOT2CSVconverter.exe
EXE=/global/cfs/cdirs/m3443/usr/xju/code/test/converter/ROOT2CSVconverter

BASENAME=`basename $FNAME`
TAG=$(echo $BASENAME | sed 's/\./ /g' | sed 's/_/ /g' | awk '{print $16}')

echo Running $TAG
#source /global/cfs/cdirs/atlas/scripts/setupATLAS.sh && setupATLAS -c centos7+batch
#source /global/cfs/cdirs/m3443/usr/xju/code/athena/Tracking/TrkDumpAlgs/scripts/setup_cvmfs_lcg97.sh

if [ -d $TAG ]; then
	echo "$TAG is there, skip."
	continue
fi

mkdir $TAG
cd $TAG
ln -s $FNAME Dump_GNN4Itk.root

$EXE

echo $TAG is DONE
