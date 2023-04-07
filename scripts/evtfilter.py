### simple event filter joboptions file ###
## https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/PyAthenaSimpleEventSelection

import AthenaCommon.Constants as Lvl
from AthenaCommon.AppMgr import theApp
from AthenaCommon.AppMgr import ServiceMgr as svcMgr

# get a handle on the job main sequence
from AthenaCommon.AlgSequence import AlgSequence, AthSequencer
job = AlgSequence()

## event related parameters: 
##  run by default on 10 events from input file
if not 'EVTMAX' in dir():
    EVTMAX = 20

theApp.EvtMax = EVTMAX

from AthenaCommon.AthenaCommonFlags import athenaCommonFlags
athenaCommonFlags.FilesInput = ["forXiangyang/fullEvent/RDO.24294355._000052.pool.root.1"]


from AthenaCommon.GlobalFlags import globalflags
from RecExConfig.InputFilePeeker import inputFileSummary
globalflags.DataSource = 'data' if inputFileSummary['evt_type'][0] == "IS_DATA" else 'geant4'
globalflags.DetDescrVersion = inputFileSummary['geometry']


from InDetSLHC_Example.SLHC_JobProperties import SLHC_Flags
from InDetRecExample.InDetJobProperties import InDetFlags
include("InDetSLHC_Example/preInclude.SLHC_Setup.py")
include('InDetSLHC_Example/preInclude.SLHC_Setup_Strip_GMX.py')
include('InDetSLHC_Example/preInclude.SLHC.SiliconOnly.Reco.py')
SLHC_Flags.doGMX.set_Value_and_Lock(True)

# Just turn on the detector components we need
from AthenaCommon.DetFlags import DetFlags 

DetFlags.detdescr.all_setOff() 
DetFlags.detdescr.SCT_setOn() 
DetFlags.detdescr.BField_setOn() 
DetFlags.detdescr.pixel_setOn() 

# Set up geometry and BField 
include("RecExCond/AllDet_detDescr.py")

SLHC_Flags.SLHC_Version = ''

include('InDetSLHC_Example/postInclude.SLHC_Setup_ITK.py')

## input file
import AthenaPoolCnvSvc.ReadAthenaPool
svcMgr.EventSelector.InputCollections = ['forXiangyang/fullEvent/RDO.24294355._000052.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000054.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000056.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000058.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000063.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000064.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000065.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000066.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000067.pool.root.1',
 'forXiangyang/fullEvent/RDO.24294355._000068.pool.root.1']

## filter configuration ##
##  -> we use the special sequence 'AthFilterSeq' which
##      is run before any other algorithm (which are usually in the
##      'TopAlg' sequence
seq = AthSequencer("AthFilterSeq")

from GaudiSequencer.PyComps import PyEvtFilter
seq += PyEvtFilter(
   'alg',
   # the store-gate key. leave as an empty string to take any eventinfo instance
   evt_info='McEventInfo', 
   OutputLevel=Lvl.INFO)

# seq.alg.evt_list = [24723, 24737, 24728, 24755, 24777, 24774, 24791, 24800, 24816, 24801]
# seq.alg.evt_list = [24723, 24737, 24728]

seq.alg.evt_list = [
    24723, 24737, 24728, 24755, 24777, 24774, 24791, 24800, 24816, 24801,
    25724, 25737, 25752, 25776, 25767, 25788, 25813, 25801, 25835, 25833,
    26735, 26739, 26779, 26765, 26786, 26800, 26801, 26823, 26830, 26877,
    27654, 27659, 27672, 27676, 27680, 27688, 27703, 27739, 27754, 27779,
    30159, 30142, 30169, 30199, 30201, 30202, 30242, 30295, 30283, 30353,
    30668, 30662, 30693, 30697, 30689, 30714, 30725, 30755, 30812, 30810, 
    31162, 31161, 31170, 31194, 31186, 31213, 31228, 31239, 31256, 31278, 
    31668, 31667, 31700, 31705, 31721, 31728, 31768, 31773, 31800, 31795, 
    32178, 32170, 32182, 32185, 32212, 32209, 32227, 32257, 32249, 32283, 
    32662, 32668, 32689, 32699, 32707, 32715, 32708, 32730, 32796, 32809
]


# for the list of event numbers above, apply the following
# filter policy (ie: here we accept event numbers 1,4,5,6)
seq.alg.filter_policy = 'accept' # 'reject'

# we could have use a lambda function selecting 'even' event numbers
# NOTE: you can't use both 'evt_list' and 'filter_fct'
# NOTE: the signature of the lambda function *has* to be 'run,evt'
#seq.alg.filter_fct = lambda run,evt: evt%2 == 0

# we now add an event counter in the usual TopAlg sequence to see
# the result of the filtering above
job += CfgMgr.AthEventCounter('counter', OutputLevel=Lvl.INFO)

svcMgr.MessageSvc.OutputLevel = Lvl.ERROR

### saving the selected events ------------------------------------------------
import AthenaPoolCnvSvc.WriteAthenaPool as wap
outStream = wap.AthenaPoolOutputStream("StreamRDO", "filtered.RDO.pool.root", True)
outStream.ForceRead = True
outStream.TakeItemsFromInput = True

# tell the stream about the filtering alg
outStream.AcceptAlgs = [seq.alg.name()]

## optimization: tweak the default commit (to disk) interval time to 
##               every 100 events as mc events are fairly small
svcMgr.AthenaPoolCnvSvc.CommitInterval = 100


### EOF ###