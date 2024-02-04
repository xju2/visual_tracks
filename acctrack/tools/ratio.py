import ROOT

def create_ratio(hist_ref: ROOT.TH1, hist_comparator: ROOT.TH1):
    ratio = hist_comparator.Clone()
    ratio.Divide(hist_comparator, hist_ref, 1.0, 1.0, "B")

    ratio.SetLineColor(ROOT.kBlack)
    ratio.SetLineWidth(1)
    ratio.SetMarkerColor(ROOT.kBlack)
    ratio.SetMarkerStyle(8)

    # these are options that could change
    ratio.GetYaxis().SetTitle("Ratio")
    # take care of axes, labels for ratio
    ratio.SetTitle("")
    ratio.GetXaxis().SetLabelSize(0.15)
    ratio.GetXaxis().SetTitleSize(0.17)
    ratio.GetXaxis().SetTickLength(0.1)
    ratio.GetXaxis().SetTitleOffset(1.0)

    ratio.GetYaxis().SetLabelSize(0.14)
    ratio.GetYaxis().SetTitleSize(0.15)
    ratio.GetYaxis().SetTitleOffset(0.33)
    ratio.GetYaxis().SetNdivisions(704)

    return ratio
