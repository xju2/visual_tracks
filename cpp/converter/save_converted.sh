#!/bin/bash

IN_DIR=$1
OUT_DIR=$2
if [ $# -lt 2 ]; then
	echo "Need input dir and output dir"
	exit
fi

echo $IN_DIR
JOBS=$(cat $IN_DIR/done.txt)
for JOB in $JOBS
do
	IDX=`expr $JOB + 1`
	if [ $IDX -lt 10 ]; then
		SRC_NAME=J00${IDX}
	elif [ $IDX -lt 100 ]; then
		SRC_NAME=J0${IDX}
	else
		SRC_NAME=J${IDX}
	fi
	SRC_DIR=$IN_DIR/job${JOB}/$SRC_NAME
	if [ -d $SRC_DIR ]; then
		echo "coping $SRC_DIR to $OUT_DIR/$SRC_NAME"
		cp -r $SRC_DIR $OUT_DIR/$SRC_NAME
	fi
done
