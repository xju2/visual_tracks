#ifndef __ROOT2CSVconverter_hpp
#include "ROOT2CSVconverter.hpp"
#endif

#include <sstream>
//#pragma link C++ class std::vector<std::vector<int> >+;

//----------------------------------------------------------------
ROOT2CSVconverter::ROOT2CSVconverter (const std::string& filename)
//----------------------------------------------------------------
{
  RootFile = TFile::Open (filename.c_str(), "READ");
  if (!RootFile) throw std::invalid_argument ("Cannot open file " + filename);
  RootTree = (TTree*) RootFile -> Get ("GNN4ITk");
  _nb_events = RootTree -> GetEntries();
  std::cout << "nb of events = " << _nb_events << std::endl;
}


//--------------------------------------
ROOT2CSVconverter::~ROOT2CSVconverter ()
//--------------------------------------
{
  RootFile -> Close ();
  delete[] m_SEID;

  delete[] m_Part_event_number;
  delete[] m_Part_barcode;
  delete[] m_Part_px, m_Part_py, m_Part_pz;
  delete[] m_Part_pt;
  delete[] m_Part_eta;
  delete[] m_Part_vx, m_Part_vy, m_Part_vz;
  delete[] m_Part_radius, m_Part_status, m_Part_charge;
  delete[] m_Part_pdg_id, m_Part_passed;
  delete[] m_Part_vProdNin, m_Part_vProdNout, m_Part_vProdStatus, m_Part_vProdBarcode;
  delete m_Part_vParentID, m_Part_vParentBarcode;

  delete [] m_CLindex;
  delete m_CLhardware;
  delete [] m_CLx;
  delete [] m_CLy;
  delete [] m_CLz;
  delete [] m_CLbarrel_endcap;
  delete [] m_CLlayer_disk;
  delete [] m_CLeta_module;
  delete [] m_CLphi_module;
  delete [] m_CLside;
//  delete [] m_CLmoduleID;
  delete m_CLparticleLink_eventIndex;
  delete m_CLparticleLink_barcode;
  delete m_CLbarcodesLinked;
  delete m_CLphis, m_CLetas, m_CLtots;
  delete [] m_CLloc_direction1, m_CLloc_direction2, m_CLloc_direction3;
  delete [] m_CLJan_loc_direction1, m_CLJan_loc_direction2, m_CLJan_loc_direction3;
  delete [] m_CLpixel_count;
  delete [] m_CLcharge_count;
  delete [] m_CLloc_eta, m_CLloc_phi;
  delete [] m_CLglob_eta, m_CLglob_phi;
  delete [] m_CLeta_angle, m_CLphi_angle;
  delete [] m_CLnorm_x, m_CLnorm_y, m_CLnorm_z;
  delete m_CLlocal_cov;

  delete [] m_SPindex;
  delete [] m_SPx, m_SPy, m_SPz;
  delete [] m_SPCL1_index, m_SPCL2_index;

  delete [] m_TRKindex;
  delete [] m_TRKtrack_fitter, m_TRKparticle_hypothesis;
  delete m_TRKproperties, m_TRKpattern;
  delete [] m_TRKndof, m_TRKmot, m_TRKoot;
  delete [] m_TRKchiSq;
  delete m_TRKmeasurementsOnTrack_pixcl_sctcl_index;
  delete m_TRKoutliersOnTrack_pixcl_sctcl_index;
  delete [] m_TRKcharge;
  delete m_TRKperigee_position, m_TRKperigee_momentum;
  delete [] m_TTCindex, m_TTCevent_index, m_TTCparticle_link, m_TTCprobability;

  delete [] m_DTTindex, m_DTTsize;
  delete m_DTTtrajectory_eventindex, m_DTTtrajectory_barcode, m_DTTstTruth_subDetType, m_DTTstTrack_subDetType, m_DTTstCommon_subDetType;
}


//------------------------------------------
void ROOT2CSVconverter::convert_subevents ()
//------------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nSE", &m_nSE);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nSE) max_size = m_nSE;
  }

  m_SEID = new int[max_size];
  RootTree -> SetBranchAddress("SEID", m_SEID);

  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "subevents_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());
    for (int i=0; i<m_nSE; i++)
      file << m_SEID[i] << std::endl;
    file.close();
  }
  std::cout << "subevents_evt*.dat files generated" << std::endl;
}


//------------------------------------------
void ROOT2CSVconverter::convert_particles ()
//------------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nPartEVT", &m_nPartEVT);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nPartEVT) max_size = m_nPartEVT;
  }

  // allocates memory
  m_Part_event_number = new int[max_size];
  RootTree -> SetBranchAddress("Part_event_number", m_Part_event_number);
  m_Part_barcode = new int[max_size];
  RootTree -> SetBranchAddress("Part_barcode", m_Part_barcode);
  m_Part_px = new float[max_size];
  RootTree -> SetBranchAddress("Part_px", m_Part_px);
  m_Part_py = new float[max_size];
  RootTree -> SetBranchAddress("Part_py", m_Part_py);
  m_Part_pz = new float[max_size];
  RootTree -> SetBranchAddress("Part_pz", m_Part_pz);
  m_Part_pt = new float[max_size];
  RootTree -> SetBranchAddress("Part_pt", m_Part_pt);
  m_Part_eta = new float[max_size];
  RootTree -> SetBranchAddress("Part_eta", m_Part_eta);
  m_Part_vx = new float[max_size];
  RootTree -> SetBranchAddress("Part_vx", m_Part_vx);
  m_Part_vy = new float[max_size];
  RootTree -> SetBranchAddress("Part_vy", m_Part_vy);
  m_Part_vz = new float[max_size];
  RootTree -> SetBranchAddress("Part_vz", m_Part_vz);
  m_Part_radius = new float[max_size];
  RootTree -> SetBranchAddress("Part_radius", m_Part_radius);
  m_Part_status = new float[max_size];
  RootTree -> SetBranchAddress("Part_status", m_Part_status);
  m_Part_charge = new float[max_size];
  RootTree -> SetBranchAddress("Part_charge", m_Part_charge);
  m_Part_pdg_id = new int[max_size];
  RootTree -> SetBranchAddress("Part_pdg_id", m_Part_pdg_id);
  m_Part_passed = new int[max_size];
  RootTree -> SetBranchAddress("Part_passed", m_Part_passed);
  m_Part_vProdNin = new int[max_size];
  RootTree -> SetBranchAddress("Part_vProdNin", m_Part_vProdNin);
  m_Part_vProdNout = new int[max_size];
  RootTree -> SetBranchAddress("Part_vProdNout", m_Part_vProdNout);
  m_Part_vProdStatus = new int[max_size];
  RootTree -> SetBranchAddress("Part_vProdStatus", m_Part_vProdStatus);
  m_Part_vProdBarcode = new int[max_size];
  RootTree -> SetBranchAddress("Part_vProdBarcode", m_Part_vProdBarcode);
  m_Part_vParentID = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress("Part_vParentID", &m_Part_vParentID);
  m_Part_vParentBarcode = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress("Part_vParentBarcode", &m_Part_vParentBarcode);

  // read branch and save file in csv format
  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "particles_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());
    for (int i=0; i<m_nPartEVT; i++) {
      std::string passed = (m_Part_passed[i]) ? "YES": "NO";
      file << m_Part_event_number[i] << "," << m_Part_barcode[i] << "," << m_Part_px[i] << "," << m_Part_py[i] << "," << m_Part_pz[i] << "," << m_Part_pt[i] << "," << m_Part_eta[i] << "," << m_Part_vx[i] << "," << m_Part_vy[i] << "," << m_Part_vz[i] << ","
           << m_Part_radius[i] << "," << m_Part_status[i] << "," << m_Part_charge[i] << "," << m_Part_pdg_id[i] << "," << passed << "," << m_Part_vProdNin[i] << "," << m_Part_vProdNout[i] << "," << m_Part_vProdStatus[i] << "," << m_Part_vProdBarcode[i] << ",#,";
      if ((*m_Part_vParentID)[i].size()) {
        for (int j=0; j<(*m_Part_vParentID)[i].size(); j++)
          file << "(" << (*m_Part_vParentID)[i][j] << "," << (*m_Part_vParentBarcode)[i][j] << "),";
      }
      file << "#";
      file << std::endl;
    }
    file.close();
  }
  std::cout << "particles_evt*.dat files generated" << std::endl;
}


//-----------------------------------------
void ROOT2CSVconverter::convert_clusters ()
//-----------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nCL", &m_nCL);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nCL) max_size = m_nCL;
  }

  // allocates memory
  m_CLindex = new int[max_size];
  RootTree -> SetBranchAddress ("CLindex", m_CLindex);
  m_CLhardware = new std::vector<std::string>;
  RootTree -> SetBranchAddress ("CLhardware", &m_CLhardware);
  m_CLx = new double[max_size];
  RootTree -> SetBranchAddress ("CLx", m_CLx);
  m_CLy = new double[max_size];
  RootTree -> SetBranchAddress ("CLy", m_CLy);
  m_CLz = new double[max_size];
  RootTree -> SetBranchAddress ("CLz", m_CLz);
  m_CLbarrel_endcap = new int[max_size];
  RootTree -> SetBranchAddress ("CLbarrel_endcap", m_CLbarrel_endcap);
  m_CLlayer_disk = new int[max_size];
  RootTree -> SetBranchAddress ("CLlayer_disk", m_CLlayer_disk);
  m_CLeta_module = new int[max_size];
  RootTree -> SetBranchAddress ("CLeta_module", m_CLeta_module);
  m_CLphi_module = new int[max_size];
  RootTree -> SetBranchAddress ("CLphi_module", m_CLphi_module);
  m_CLside = new int[max_size];
  RootTree -> SetBranchAddress ("CLside", m_CLside);
//  m_CLmoduleID = new uint64_t[max_size];
//  RootTree -> SetBranchAddress ("CLmoduleID", m_CLmoduleID);
  m_CLparticleLink_eventIndex = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("CLparticleLink_eventIndex", &m_CLparticleLink_eventIndex);
  m_CLparticleLink_barcode = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("CLparticleLink_barcode", &m_CLparticleLink_barcode);
  m_CLbarcodesLinked = new std::vector<std::vector<bool>>;
  RootTree -> SetBranchAddress ("CLbarcodesLinked", &m_CLbarcodesLinked);
  m_CLphis = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("CLphis", &m_CLphis);
  m_CLetas = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("CLetas", &m_CLetas);
  m_CLtots = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("CLtots", &m_CLtots);
  m_CLloc_direction1 = new double[max_size];
  RootTree -> SetBranchAddress ("CLloc_direction1", m_CLloc_direction1);
  m_CLloc_direction2 = new double[max_size];
  RootTree -> SetBranchAddress ("CLloc_direction2", m_CLloc_direction2);
  m_CLloc_direction3 = new double[max_size];
  RootTree -> SetBranchAddress ("CLloc_direction3", m_CLloc_direction3);
  m_CLJan_loc_direction1 = new double[max_size];
  RootTree -> SetBranchAddress ("CLJan_loc_direction1", m_CLJan_loc_direction1);
  m_CLJan_loc_direction2 = new double[max_size];
  RootTree -> SetBranchAddress ("CLJan_loc_direction2", m_CLJan_loc_direction2);
  m_CLJan_loc_direction3 = new double[max_size];
  RootTree -> SetBranchAddress ("CLJan_loc_direction3", m_CLJan_loc_direction3);
  m_CLpixel_count = new int[max_size];
  RootTree -> SetBranchAddress ("CLpixel_count", m_CLpixel_count);
  m_CLcharge_count = new float[max_size];
  RootTree -> SetBranchAddress ("CLcharge_count", m_CLcharge_count);
  m_CLloc_eta = new float[max_size];
  RootTree -> SetBranchAddress ("CLloc_eta", m_CLloc_eta);
  m_CLloc_phi = new float[max_size];
  RootTree -> SetBranchAddress ("CLloc_phi", m_CLloc_phi);
  m_CLglob_eta = new float[max_size];
  RootTree -> SetBranchAddress ("CLglob_eta", m_CLglob_eta);
  m_CLglob_phi = new float[max_size];
  RootTree -> SetBranchAddress ("CLglob_phi", m_CLglob_phi);
  m_CLeta_angle = new double[max_size];
  RootTree -> SetBranchAddress ("CLeta_angle", m_CLeta_angle);
  m_CLphi_angle = new double[max_size];
  RootTree -> SetBranchAddress ("CLphi_angle", m_CLphi_angle);
  m_CLnorm_x = new float[max_size];
  RootTree -> SetBranchAddress ("CLnorm_x", m_CLnorm_x);
  m_CLnorm_y = new float[max_size];
  RootTree -> SetBranchAddress ("CLnorm_y", m_CLnorm_y);
  m_CLnorm_z = new float[max_size];
  RootTree -> SetBranchAddress ("CLnorm_z", m_CLnorm_z);
  m_CLlocal_cov = new std::vector<std::vector<double>>;
  RootTree -> SetBranchAddress ("CLlocal_cov", &m_CLlocal_cov);

  // read branch and save file in csv format
  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "clusters_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());

    for (int i=0; i<m_nCL; i++) {
      file << m_CLindex[i] << "," << (*m_CLhardware)[i] << "," << m_CLx[i] << "," << m_CLy[i] << "," << m_CLz[i] << "," << "#," << m_CLbarrel_endcap[i] << "," << m_CLlayer_disk[i] << "," << m_CLeta_module[i] << "," << m_CLphi_module[i] << "," << m_CLside[i] << ",#,";
      for (int j=0; j<(*m_CLparticleLink_eventIndex)[i].size(); j++)
        file << "(" << (*m_CLparticleLink_eventIndex)[i][j] << "," << (*m_CLparticleLink_barcode)[i][j] << "," << (*m_CLbarcodesLinked)[i][j] << "),";
      file << "#,";
      for (int j=0; j<(*m_CLtots)[i].size(); j++)
        file << "(" << (*m_CLetas)[i][j] << "," << (*m_CLphis)[i][j] << "," << (*m_CLtots)[i][j] << "),";
      file << "#," << m_CLpixel_count[i] << "," << m_CLcharge_count[i] << "," << m_CLloc_eta[i] << "," << m_CLloc_phi[i] << "," << m_CLloc_direction1[i] << "," << m_CLloc_direction2[i] << "," << m_CLloc_direction3[i] << "," << m_CLJan_loc_direction1[i] << "," << m_CLJan_loc_direction2[i] << "," << m_CLJan_loc_direction3[i] << ","
           << m_CLglob_eta[i] << "," << m_CLglob_phi[i] << "," << m_CLeta_angle[i] << "," << m_CLphi_angle[i] << ",#," <<m_CLnorm_x[i] << "," << m_CLnorm_y[i] << "," << m_CLnorm_z[i] << ",#";
      for (int j=0; j<(*m_CLlocal_cov)[i].size(); j++)
        file << "," << (*m_CLlocal_cov)[i][j];
      file << std::endl;
    }
  }
  std::cout << "clusters_evt*.dat files generated" << std::endl;
}


//---------------------------------------------
void ROOT2CSVconverter::convert_space_points ()
//---------------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nSP", &m_nSP);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nSP) max_size = m_nSP;
  }

  // allocates memory
  m_SPindex = new int[max_size];
  RootTree -> SetBranchAddress ("SPindex", m_SPindex);
  m_SPx = new double[max_size];
  RootTree -> SetBranchAddress ("SPx", m_SPx);
  m_SPy = new double[max_size];
  RootTree -> SetBranchAddress ("SPy", m_SPy);
  m_SPz = new double[max_size];
  RootTree -> SetBranchAddress ("SPz", m_SPz);
  m_SPCL1_index = new int[max_size];
  RootTree -> SetBranchAddress ("SPCL1_index", m_SPCL1_index);
  m_SPCL2_index = new int[max_size];
  RootTree -> SetBranchAddress ("SPCL2_index", m_SPCL2_index);

  // read branch and save file in csv format
  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "spacepoints_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());

    for (int i=0; i<m_nSP; i++) {
      file << m_SPindex[i] << "," << m_SPx[i] << "," << m_SPy[i] << "," << m_SPz[i] << "," << m_SPCL1_index[i];
      if (m_SPCL2_index[i] != -1) file << "," << m_SPCL2_index[i];
      file << std::endl;
    }
  }
  std::cout << "spacepoints_evt*.dat files generated" << std::endl;
}


//---------------------------------------
void ROOT2CSVconverter::convert_tracks ()
//---------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nTRK", &m_nTRK);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nTRK) max_size = m_nTRK;
  }

  // allocates memory
  m_TRKindex = new int[max_size];
  RootTree -> SetBranchAddress ("TRKindex", m_TRKindex);
  m_TRKtrack_fitter = new int[max_size];
  RootTree -> SetBranchAddress ("TRKtrack_fitter", m_TRKtrack_fitter);
  m_TRKparticle_hypothesis = new int[max_size];
  RootTree -> SetBranchAddress ("TRKparticle_hypothesis", m_TRKparticle_hypothesis);
  m_TRKproperties = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("TRKproperties", &m_TRKproperties);
  m_TRKpattern = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("TRKpattern", &m_TRKpattern);
  m_TRKndof = new int[max_size];
  RootTree -> SetBranchAddress ("TRKndof", m_TRKndof);
  m_TRKmot = new int[max_size];
  RootTree -> SetBranchAddress ("TRKmot", m_TRKmot);
  m_TRKoot = new int[max_size];
  RootTree -> SetBranchAddress ("TRKoot", m_TRKoot);
  m_TRKchiSq = new float[max_size];
  RootTree -> SetBranchAddress ("TRKchiSq", m_TRKchiSq);
  m_TRKmeasurementsOnTrack_pixcl_sctcl_index = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("TRKmeasurementsOnTrack_pixcl_sctcl_index", &m_TRKmeasurementsOnTrack_pixcl_sctcl_index);
  m_TRKoutliersOnTrack_pixcl_sctcl_index = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("TRKoutliersOnTrack_pixcl_sctcl_index", &m_TRKoutliersOnTrack_pixcl_sctcl_index);
  m_TRKcharge = new int[max_size];
  RootTree -> SetBranchAddress ("TRKcharge", m_TRKcharge);
  m_TRKperigee_position = new std::vector<std::vector<double>>;
  RootTree -> SetBranchAddress ("TRKperigee_position", &m_TRKperigee_position);
  m_TRKperigee_momentum = new std::vector<std::vector<double>>;
  RootTree -> SetBranchAddress ("TRKperigee_momentum", &m_TRKperigee_momentum);
  m_TTCindex = new int[max_size];
  RootTree -> SetBranchAddress ("TTCindex", m_TTCindex);
  m_TTCevent_index = new int[max_size];
  RootTree -> SetBranchAddress ("TTCevent_index", m_TTCevent_index);
  m_TTCparticle_link = new int[max_size];
  RootTree -> SetBranchAddress ("TTCparticle_link", m_TTCparticle_link);
  m_TTCprobability = new float[max_size];
  RootTree -> SetBranchAddress ("TTCprobability", m_TTCprobability);

  // read branch and save file in csv format
  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "tracks_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());

    for (int i=0; i<m_nTRK; i++) {
      file << m_TRKindex[i] << "," << m_TRKtrack_fitter[i] << "," << m_TRKparticle_hypothesis[i] << ",#,";
      for (int j=0; j<(*m_TRKproperties)[i].size(); j++)
        file << (*m_TRKproperties)[i][j] << ",";
      file << "#,#,";
      for (int j=0; j<(*m_TRKpattern)[i].size(); j++)
        file << (*m_TRKpattern)[i][j] << ",";
      file << "#," << m_TRKndof[i] << "," << m_TRKchiSq[i] << ",";
      if (((*m_TRKperigee_position)[i]).size()) {
        file << m_TRKcharge[i] << ",";
        for (int j=0; j<(*m_TRKperigee_position)[i].size(); j++)
          file << (*m_TRKperigee_position)[i][j] << ",";
        for (int j=0; j<(*m_TRKperigee_momentum)[i].size(); j++)
          file << (*m_TRKperigee_momentum)[i][j] << ",";
      }
      file << m_TRKmot[i] << "," << m_TRKoot[i] << ",#,";
      for (int j=0; j<(*m_TRKmeasurementsOnTrack_pixcl_sctcl_index)[i].size(); j++)
        file << (*m_TRKmeasurementsOnTrack_pixcl_sctcl_index)[i][j] << ",";
      file << "#,";
      for (int j=0; j<(*m_TRKoutliersOnTrack_pixcl_sctcl_index)[i].size(); j++)
        file << (*m_TRKoutliersOnTrack_pixcl_sctcl_index)[i][j] << ",";
      file << "#," << m_TTCindex[i] << "," << m_TTCevent_index[i] << "," << m_TTCparticle_link[i] << "," << m_TTCprobability[i];
      file << std::endl;
    }
  }
  std::cout << "tracks_evt*.dat files generated" << std::endl;

}


//-----------------------------------------------------
void ROOT2CSVconverter::convert_detailed_track_truth ()
//-----------------------------------------------------
{
  int max_size = 0;

  RootTree -> SetBranchAddress("nDTT", &m_nDTT);

  for (int i=0; i<_nb_events; i++) {
    RootTree -> GetEntry (i);
    if (max_size < m_nDTT) max_size = m_nDTT;
  }

  // allocates memory
  m_DTTindex = new int[max_size];
  RootTree -> SetBranchAddress ("DTTindex", m_DTTindex);
  m_DTTsize = new int[max_size];
  RootTree -> SetBranchAddress ("DTTsize", m_DTTsize);
  m_DTTtrajectory_eventindex = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("DTTtrajectory_eventindex", &m_DTTtrajectory_eventindex);
  m_DTTtrajectory_barcode = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("DTTtrajectory_barcode", &m_DTTtrajectory_barcode);
  m_DTTstTruth_subDetType = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("DTTstTruth_subDetType", &m_DTTstTruth_subDetType);
  m_DTTstTrack_subDetType = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("DTTstTrack_subDetType", &m_DTTstTrack_subDetType);
  m_DTTstCommon_subDetType = new std::vector<std::vector<int>>;
  RootTree -> SetBranchAddress ("DTTstCommon_subDetType",&m_DTTstCommon_subDetType);

  // read branch and save file in csv format
  for (int n=0; n<_nb_events; n++) {
    RootTree -> GetEntry (n);
    std::stringstream ss;
    ss << n+1;
    std::string filename = "detailedtracktruth_evt" +ss.str() + ".dat";

    std::ofstream file (filename.c_str(), std::ios::out);
    assert (!file.fail());

    for (int i=0; i<m_nDTT; i++) {
      file << m_DTTindex[i] << "," << m_DTTsize[i] << ",#,";
      for (int j=0; j<(*m_DTTtrajectory_eventindex)[i].size(); j++)
        file << (*m_DTTtrajectory_eventindex)[i][j] << "," << (*m_DTTtrajectory_barcode)[i][j] << ",";
      file << "#";
      for (int j=0; j<(*m_DTTstTruth_subDetType)[i].size(); j++)
        file << "," << (*m_DTTstTruth_subDetType)[i][j];
      for (int j=0; j<(*m_DTTstTrack_subDetType)[i].size(); j++)
        file << "," << (*m_DTTstTrack_subDetType)[i][j];
      for (int j=0; j<(*m_DTTstCommon_subDetType)[i].size(); j++)
        file << "," << (*m_DTTstCommon_subDetType)[i][j];
      file << std::endl;
    }
  }

  std::cout << "detailedtracktruth_evt*.dat files generated" << std::endl;
}


//========
int main()
//========
{
  std::cout << "Starting convertion ROOT -> txt" << std::endl;

  std::string filename = "Dump_GNN4Itk.root";
  ROOT2CSVconverter converter (filename);

  converter.convert_subevents ();
  converter.convert_particles ();
  converter.convert_clusters ();
  converter.convert_space_points ();
  // converter.convert_tracks ();
  // converter.convert_detailed_track_truth ();
}
