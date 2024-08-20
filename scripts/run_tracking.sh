#!/bin/bash

# RDO_FILENAME="../run_main_clean/reduced_RDO.r14701.33629030._000008.pool.root.1"
RDO_FILENAME="inputData/RDO.37737772._000213.pool.root.1"

function gnn_tracking() {
    rm InDetIdDict.xml PoolFileCatalog.xml hostnamelookup.tmp eventLoopHeartBeat.txt
    export ATHENA_CORE_NUMBER=1
    # --skipEvents 44
    # 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude'
    # --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; from AthOnnxComps.OnnxRuntimeFlags import OnnxRuntimeType; flags.AthOnnx.ExecutionProvider = OnnxRuntimeType.CUDA' \
    # --postExec 'all:cfg.getService("AlgResourcePool").CountAlgorithmInstanceMisses = True' \
    # --perfmon 'fullmonmt' \

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'False' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier,InDetConfig.SiSpacePointFormationConfig.InDetToXAODSpacePointConversionCfg' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnTritonValidation' \
        --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; flags.Tracking.GNN.Triton.model = "ExatrkX4PixelPythonWithFilter"; flags.Tracking.GNN.Triton.url = "nid200256"; flags.Tracking.GNN.DumpObjects.doDetailedTracksInfo = True; flags.Tracking.GNN.DumpObjects.doClusters = True; flags.Tracking.GNN.DumpObjects.doParticles = True' \
        --postExec 'from InDetGNNTracking.InDetGNNTrackingConfig import DumpObjectsCfg; cfg.merge(DumpObjectsCfg(flags)); msg=cfg.getService("MessageSvc"); msg.infoLimit = 9999999; msg.debugLimit = 9999999; msg.verboseLimit = 9999999;' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.gnnreader.debug.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=DEBUG' \
        --maxEvents 1 2>&1 | tee log.gnnreader_debug.txt
}

function ckf_tracking() {
    rm InDetIdDict.xml PoolFileCatalog.xml hostnamelookup.tmp eventLoopHeartBeat.txt
    export ATHENA_CORE_NUMBER=1
    # --skipEvents 44
    # 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude'
    # --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; from AthOnnxComps.OnnxRuntimeFlags import OnnxRuntimeType; flags.AthOnnx.ExecutionProvider = OnnxRuntimeType.CUDA' \
    # --postExec 'all:cfg.getService("AlgResourcePool").CountAlgorithmInstanceMisses = True' \
    # --perfmon 'fullmonmt' \
        # --preExec 'flags.Tracking.GNN.DumpObjects.doDetailedTracksInfo = True; flags.Tracking.GNN.DumpObjects.doClusters = False; flags.Tracking.GNN.DumpObjects.doParticles = False' \
        # --postExec 'from InDetGNNTracking.InDetGNNTrackingConfig import DumpObjectsCfg; cfg.merge(DumpObjectsCfg(flags))' \

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'False' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier,InDetConfig.SiSpacePointFormationConfig.InDetToXAODSpacePointConversionCfg' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' \
        --preExec 'flags.Tracking.GNN.DumpObjects.doDetailedTracksInfo = True; flags.Tracking.GNN.DumpObjects.doClusters = True; flags.Tracking.GNN.DumpObjects.doParticles = True' \
        --postExec 'from InDetGNNTracking.InDetGNNTrackingConfig import DumpObjectsCfg; cfg.merge(DumpObjectsCfg(flags))' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.ckf.debug.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=INFO' \
        --maxEvents 1 2>&1 | tee log.ckf.txt
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

ckf_tracking
