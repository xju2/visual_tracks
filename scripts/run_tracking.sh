#!/bin/bash

# RDO_FILENAME="../run_main_clean/reduced_RDO.r14701.33629030._000008.pool.root.1"
RDO_FILENAME="inputData/RDO.37737772._000213.pool.root.1"

function gnn_tracking() {
    rm InDetIdDict.xml PoolFileCatalog.xml
    # export ATHENA_CORE_NUMBER=6
    # --skipEvents 44
    # 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude'
    # --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; from AthOnnxComps.OnnxRuntimeFlags import OnnxRuntimeType; flags.AthOnnx.ExecutionProvider = OnnxRuntimeType.CUDA' \

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'False' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnTritonValidation' \
        --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.gnnreader.debug.root'  \
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


gnn_tracking
