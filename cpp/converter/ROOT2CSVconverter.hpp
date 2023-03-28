#ifndef __ROOT2CSV__
#define __ROOT2CSV__
#ifndef __cplusplus
#error Must use C++ for the type vector
#endif

#ifndef __iostream
#include <iostream>
#endif

#ifndef __fstream
#include <fstream>
#endif

#ifndef __assert_h
#include <assert.h>
#endif

#ifndef __TFile_h
#include <TFile.h>
#endif

#ifndef __TTree_h
#include <TTree.h>
#endif

#ifndef __TROOT_H
#include <TROOT.h>
#endif

//=====================
class ROOT2CSVconverter
//=====================
{
  private:
    int _nb_events;
    TFile* RootFile;
    TTree* RootTree;

    int m_nSE;
    int *m_SEID;

    int m_nPartEVT;
    int *m_Part_event_number;
    int *m_Part_barcode;
    float *m_Part_px, *m_Part_py, *m_Part_pz;
    float *m_Part_pt;
    float *m_Part_eta;
    float *m_Part_vx, *m_Part_vy, *m_Part_vz;
    float *m_Part_radius, *m_Part_status, *m_Part_charge;
    int *m_Part_pdg_id, *m_Part_passed;
    int *m_Part_vProdNin, *m_Part_vProdNout, *m_Part_vProdStatus, *m_Part_vProdBarcode;
    std::vector<std::vector<int>> *m_Part_vParentID, *m_Part_vParentBarcode;

    int m_nCL;
    int *m_CLindex;
    std::vector<std::string> *m_CLhardware;
    double *m_CLx, *m_CLy, *m_CLz;
    int *m_CLbarrel_endcap;
    int *m_CLlayer_disk;
    int *m_CLeta_module;
    int *m_CLphi_module;
    int *m_CLside;
//    uint64_t *m_CLmoduleID;
    std::vector<std::vector<int>> *m_CLparticleLink_eventIndex;
    std::vector<std::vector<int>> *m_CLparticleLink_barcode;
    std::vector<std::vector<bool>> *m_CLbarcodesLinked;
    std::vector<std::vector<int>> *m_CLphis, *m_CLetas, *m_CLtots;
    double *m_CLloc_direction1, *m_CLloc_direction2,*m_CLloc_direction3;
    double *m_CLJan_loc_direction1, *m_CLJan_loc_direction2, *m_CLJan_loc_direction3;
    int *m_CLpixel_count;
    float *m_CLcharge_count;
    float *m_CLloc_eta, *m_CLloc_phi;
    float *m_CLglob_eta, *m_CLglob_phi;
    double *m_CLeta_angle, *m_CLphi_angle;
    float *m_CLnorm_x, *m_CLnorm_y, *m_CLnorm_z;
    std::vector<std::vector<double>> *m_CLlocal_cov;

    int m_nSP;
    int* m_SPindex;
    double *m_SPx, *m_SPy, *m_SPz;
    int *m_SPCL1_index, *m_SPCL2_index;

    int m_nTRK;
    int *m_TRKindex;
    int *m_TRKtrack_fitter, *m_TRKparticle_hypothesis;
    std::vector<std::vector<int>> *m_TRKproperties, *m_TRKpattern;
    int *m_TRKndof, *m_TRKmot, *m_TRKoot;
    float *m_TRKchiSq;
    std::vector<std::vector<int>> *m_TRKmeasurementsOnTrack_pixcl_sctcl_index, *m_TRKoutliersOnTrack_pixcl_sctcl_index;
    int *m_TRKcharge;
    std::vector<std::vector<double>> *m_TRKperigee_position, *m_TRKperigee_momentum;
    int *m_TTCindex, *m_TTCevent_index, *m_TTCparticle_link;
    float *m_TTCprobability;

    int m_nDTT;
    int *m_DTTindex, *m_DTTsize;
    std::vector<std::vector<int>> *m_DTTtrajectory_eventindex, *m_DTTtrajectory_barcode, *m_DTTstTruth_subDetType, *m_DTTstTrack_subDetType, *m_DTTstCommon_subDetType;


  public:
    ROOT2CSVconverter (const std::string&);
    ~ROOT2CSVconverter ();

    inline int nb_events() {return _nb_events;}
    void convert_subevents ();
    void convert_particles ();
    void convert_clusters ();
    void convert_space_points ();
    void convert_tracks ();
    void convert_detailed_track_truth ();
};

#endif
