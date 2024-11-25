#!/bin/bash

# RDO_FILENAME="../run_main_clean/reduced_RDO.r14701.33629030._000008.pool.root.1"
#RDO_FILENAME="inputData/RDO.37737772._000213.pool.root.1"

## release 24 ttbar events for evaluation
RDO_FILENAME="/global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/rel24_ttbar_testing/RDO.37737772._000213.pool.root.1"

## datasets for generality studies
ResultDIR="Results_20241118"
MuonPU0=$(\ls /global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/mc21_14TeV.900495.PG_single_muonpm_Pt10_etaFlatnp0_43.recon.RDO.e8481_s4149_r14697/* | paste -sd ',')
#ZmumuPU0=$(\ls /global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/mc21_14TeV.601190.PhPy8EG_AZNLO_Zmumu.recon.RDO.e8481_s4203_r14697/* | paste -sd ',')
ZmumuPU200=$(\ls /global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/mc21_14TeV.601190.PhPy8EG_AZNLO_Zmumu.recon.RDO.e8481_s4203_r14701/* | paste -sd ',')
HaaPU0=$(\ls /global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/mc21_14TeV.603008.PhPy8_ggH_H125_a55a55_4b_ctau100.recon.RDO.e8481_s4149_r14697/* | paste -sd ',')
HaaPU200=$(\ls /global/cfs/cdirs/m3443/data/GNN4ITK/RDOFiless/mc21_14TeV.603008.PhPy8_ggH_H125_a55a55_4b_ctau100.recon.RDO.e8481_s4149_r14700/* | head -1 | paste -sd ',')

echo $MuonPU0

TritionServer="nid008325"


#echo $RDO_FILENAME

function clean_up() {
    rm InDetIdDict.xml PoolFileCatalog.xml hostnamelookup.tmp eventLoopHeartBeat.txt
}

function gnn4pixel_with_dumping() {
    clean_up

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
        --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; flags.Tracking.GNN.Triton.model = "GNN4Pixel"; flags.Tracking.GNN.Triton.url = "nid200364"; flags.Tracking.GNN.DumpObjects.doDetailedTracksInfo = True; flags.Tracking.GNN.DumpObjects.doClusters = True; flags.Tracking.GNN.DumpObjects.doParticles = True' \
        --postExec 'from InDetGNNTracking.InDetGNNTrackingConfig import DumpObjectsCfg; cfg.merge(DumpObjectsCfg(flags)); msg=cfg.getService("MessageSvc"); msg.infoLimit = 9999999; msg.debugLimit = 9999999; msg.verboseLimit = 9999999;' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.gnnreader.debug.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=DEBUG' \
        --maxEvents 1 2>&1 | tee log.gnnreader_debug.txt
}

function gnn4pixel() {
    clean_up
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
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnTritonValidation' \
        --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; flags.Tracking.ITkGNNPass.doAmbiguityResolutionForGNN = False; flags.Tracking.GNN.Triton.model = "GNN4Pixel"; flags.Tracking.GNN.Triton.url = "nid200288";' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.gnnTriton.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=INFO' \
        --maxEvents -1 2>&1 | tee log.gnnTrition.txt
}

function gnn4pixel_with_AR() {
    # GNN tracking with ambiguity resolution.
    TritionServer="nid200381"
    clean_up
    export ATHENA_CORE_NUMBER=1

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'False' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnTritonValidation' \
        --preExec 'flags.Tracking.GNN.usePixelHitsOnly = True; flags.Tracking.GNN.Triton.model = "GNN4Pixel"; flags.Tracking.ITkGNNPass.doAmbiguityResolutionForGNN = True' "flags.Tracking.GNN.Triton.url = \"${TritionServer}\"" \
        --postExec 'msg=cfg.getService("MessageSvc"); msg.infoLimit = 9999999; msg.debugLimit = 9999999; msg.verboseLimit = 9999999;' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.gnnTrition_with_AR.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=DEBUG' \
        --maxEvents 1 2>&1 | tee log.gnnTrition_with_AR.txt
}


function ckf_tracking_with_dumping() {
    clean_up
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
        --maxEvents -1 2>&1 | tee log.ckf.txt
}

function ckf_tracking() {
    clean_up
    export ATHENA_CORE_NUMBER=20

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'True' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.ckf.debug.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=INFO' \
        --maxEvents -1 2>&1 | tee log.ckf.txt
}

# 'HardScatter', 'All', 'PileUp'
function run_idpvm() {
    		# --doTracksInJets  \
            # --doTechnicalEfficiency \

	IN_FILENAME=$1
	OUT_FILENAME=$2
    clean_up
	runIDPVM.py --filesInput ${IN_FILENAME} \
		--outputFile ${OUT_FILENAME} \
		--doTightPrimary \
		--doLoose \
        --truthMinPt 1000 \
		--HSFlag "All"  2>&1 | tee log.idpvm.txt
}


function metric_learning_tracking() {
    clean_up
    export ATHENA_CORE_NUMBER=1
    ModelName="MetricLearning"

    if [ "$#" -ne 4 ]; then
        echo "Usage: $0 <input_file> <out_file> <log_file> <process_name>"
        return 1
    fi

	INPUT_FILE=$1
    OUT_FILE=$2
    LOG_FILE=$3
    TAG_NAME=$4


    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'True' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnTritonValidation' \
        --preExec "flags.Tracking.GNN.Triton.model = \"${ModelName}\"; flags.Tracking.GNN.Triton.url = \"${TritionServer}\"; flags.Tracking.ITkGNNPass.doAmbiguityResolutionForGNN = False;" 'flags.Tracking.GNN.TrackFinderTritonTool.FeatureNames = "r,phi,z,cluster_x_1,cluster_y_1,cluster_z_1,cluster_x_2,cluster_y_2,cluster_z_2,count_1,charge_count_1,loc_eta_1,loc_phi_1,localDir0_1,localDir1_1,localDir2_1,lengthDir0_1,lengthDir1_1,lengthDir2_1,glob_eta_1,glob_phi_1,eta_angle_1,phi_angle_1,count_2,charge_count_2,loc_eta_2,loc_phi_2,localDir0_2,localDir1_2,localDir2_2,lengthDir0_2,lengthDir1_2,lengthDir2_2,glob_eta_2,glob_phi_2,eta_angle_2,phi_angle_2,eta,cluster_r_1,cluster_phi_1,cluster_eta_1,cluster_r_2,cluster_phi_2,cluster_eta_2"' \
        --inputRDOFile "${INPUT_FILE}" \
        --outputAODFile "${OUT_FILE}"  \
        --jobNumber '1' \
		--athenaopts='--loglevel=INFO' \
        --maxEvents -1 2>&1 | tee ${LOG_FILE}

    ./clean_athena_run.sh
    mkdir -p $ResultDIR/${TAG_NAME}
    mv $OUT_FILE $LOG_FILE prmon.* perfmonmt* $ResultDIR/${TAG_NAME}/
}

function gnn_mm_tracking() {
    clean_up
    export ATHENA_CORE_NUMBER=10

    Reco_tf.py \
        --CA 'all:True' --autoConfiguration 'everything' \
        --conditionsTag 'all:OFLCOND-MC15c-SDR-14-05' \
        --geometryVersion 'all:ATLAS-P2-RUN4-03-00-00' \
        --multithreaded 'True' \
        --steering 'doRAWtoALL' \
        --digiSteeringConf 'StandardInTimeOnlyTruth' \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnReaderValidation' \
        --preExec "flags.Tracking.ITkGNNPass.doAmbiguityResolutionForGNN = False; flags.Tracking.GNN.TrackReader.inputTracksDir = \"MM_tracks_with_AMP\"; "  \
        --inputRDOFile "${RDO_FILENAME}" \
        --outputAODFile 'test.aod.MM.gnn.root'  \
        --jobNumber '1' \
		--athenaopts='--loglevel=INFO' \
        --maxEvents 1 2>&1 | tee log.MM.gnn.txt
}

function rename_acorn_outs() {
    cd MM_tracks_with_AMP
    for file in event*.txt
    do
       # tune to track_242020_eventID.csv
       eventID=$(echo $file | cut -d'/' -f 2 | cut -d"." -f 1 | sed 's/event00//')
       ln -s "$file" "track_242020_${eventID}.csv"
    done
}

# time gnn4pixel
# time gnn4pixel_with_dumping
# time gnn4pixel_with_AR

# time metric_learning_tracking $RDO_FILENAME test.aod.ML.gnnTriton.root log.ML.gnnTrition.txt Testing

## >>(dataset is on tape only) time metric_learning_tracking "$ZmumuPU0" aod.ML.ZmumuPU0.root log.ML.ZmumuPU0.txt ZmumuPU0
# >(finished) time metric_learning_tracking "$MuonPU0" aod.ML.MuonPU0.root log.ML.MuonPU0.txt MuonPU0
# >(finished) time metric_learning_tracking "$ZmumuPU200" aod.ML.ZmumuPU200.root log.ML.ZmumuPU200.txt ZmumuPU200
# >(finished) time metric_learning_tracking "$HaaPU0" aod.ML.HaaPU0.root log.ML.HaaPU0.txt HaaPU0
# >(finished) time metric_learning_tracking "$HaaPU200" aod.ML.HaaPU200.root log.ML.HaaPU200.txt HaaPU200

function run_idpvm_validation(){
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <process_name>"
        return 1
    fi
    ProcessName=$1
    run_idpvm ${ResultDIR}/${ProcessName}/aod.ML.${ProcessName}.root ${ResultDIR}/${ProcessName}/physval.root
    ./clean_athena_run.sh
    mv log.idpvm.txt $ResultDIR/${ProcessName}/
}

# run_idpvm_validation MuonPU0 \
run_idpvm_validation ZmumuPU200
# run_idpvm_validation HaaPU0 \
# run_idpvm_validation HaaPU200

# rename_acorn_outs

# gnn_mm_tracking

# time ckf_tracking

## Run IDPVM.
# run_idpvm AllEvents/ckf/test.aod.ckf.debug.root AllEvents/ckf/test.idpvm.root
# run_idpvm AllEvents/gnn/test.aod.gnnreader.root AllEvents/gnn/physval.root
# run_idpvm AllEvents/gnn/aod.noAR.50evts.root AllEvents/gnn/physval.noAR.50evts.root
# run_idpvm AllEvents/end2end_ML/test.aod.ML.gnnTriton.root AllEvents/end2end_ML/physval.root
