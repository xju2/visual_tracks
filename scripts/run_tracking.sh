#!/bin/bash

RDO_FILENAME="../run_main_clean/reduced_RDO.r14701.33629030._000008.pool.root.1"
OUTNAME_PREFIX="gnnseeding.debug"

function gnn_tracking() {
    rm InDetIdDict.xml PoolFileCatalog.xml
    # export ATHENA_CORE_NUMBER=6
    #--skipEvents 44

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'True' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnReaderValidation' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnUsePixelHitsOnly' \
        --inputRDOFile ${RDO_FILENAME} \
		--valid 'True' \
		--validationFlags 'doInDet' \
		--outputNTUP_PHYSVALFile "test.idpvm.${OUTNAME_PREFIX}.root" \
        --outputAODFile "test.aod.${OUTNAME_PREFIX}.root"  \
        --maxEvents 1  2>&1 | tee log.gnnreader_debug.txt
}

# 'HardScatter', 'All', 'PileUp' 
function run_idpvm() {
    rm InDetIdDict.xml PoolFileCatalog.xml

	IN_FILENAME=$1
	OUT_FILENAME=$2
	runIDPVM.py --filesInput ${IN_FILENAME} --outputFile ${OUT_FILENAME} \
		--doTightPrimary \
		--doTracksInJets  \
		--doTechnicalEfficiency \
		--HSFlag "HardScatter"  2>&1 | tee log.idpvm.txt
}


# gnn_tracking
run_idpvm aod.gnnseeding.debug.root idpvm_gnnseeding_debug.root
mv log.idpvm.txt log.idpvm.debug.txt

run_idpvm aod.gnnseeding.debug.xcheck.root idpvm_gnnseeding_debug_xcheck.root
mv log.idpvm.txt log.idpvm.debug.xcheck.txt
