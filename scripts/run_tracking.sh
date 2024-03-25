#!/bin/bash

RDO_FILENAME="../run_main_clean/reduced_RDO.r14701.33629030._000008.pool.root.1"

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
        --preInclude 'all:Campaigns.PhaseIIPileUp200' 'InDetConfig.ConfigurationHelpers.OnlyTrackingPreInclude' 'InDetGNNTracking.InDetGNNTrackingFlags.gnnReaderValidation' \
        --inputRDOFile ${RDO_FILENAME} \
        --outputAODFile 'test.aod.gnnreader.debug.root'  \
        --maxEvents 1  2>&1 | tee log.gnnreader_debug.txt
}

gnn_tracking
