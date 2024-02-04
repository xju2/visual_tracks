import ROOT
import os

ROOT.gROOT.SetBatch(True)
if not ROOT.gROOT.GetColor(9000):
    mplBlue = ROOT.TColor(9000, 31 / 255, 119 / 255, 180 / 255, "mplBlue")

if not ROOT.gROOT.GetColor(9001):
    mplOrange = ROOT.TColor(9001, 1, 127 / 255, 14 / 255, "mplOrange")

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT.gROOT.LoadMacro(script_dir + "/AtlasStyle.C")
ROOT.SetAtlasStyle()
