#!/usr/bin/env python
import ROOT
from array import array
import math


def add_text(x, y, color, text, size=0.05, font=42):
    l = ROOT.TLatex()  # noqa
    l.SetTextSize(size)
    l.SetNDC()
    l.SetTextColor(color)
    l.SetTextFont(font)
    l.DrawLatex(x, y, text)


def add_line(hist, y_val, color=1, style=2, option="x"):
    x_low = hist.GetBinLowEdge(hist.GetXaxis().GetFirst())
    x_hi = hist.GetBinLowEdge(hist.GetXaxis().GetLast() + 1)
    y_low = hist.GetBinLowEdge(hist.GetYaxis().GetFirst())
    y_hi = hist.GetBinLowEdge(hist.GetYaxis().GetLast() + 1)
    line = ROOT.TLine()
    line.SetLineColor(color)
    line.SetLineStyle(style)
    line.SetLineWidth(2)
    if option.lower() == "x":
        line.DrawLine(x_low, y_val, x_hi, y_val)
    else:
        line.DrawLine(y_val, y_low, y_val, y_hi)


def make_legend(x1, y1, x2, y2):
    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.04)
    return legend


def add_band(hist, center, width, add_stats=True):
    x = array("d")
    y = array("d")
    up = array("d")
    down = array("d")
    weight = hist.Integral() / hist.GetEntries()
    for i in range(hist.GetXaxis().GetNbins()):
        ibin = i + 1
        if add_stats:
            content = hist.GetBinContent(ibin) / weight
            width = math.sqrt(width**2 + 1.0 / content)

        x.append(hist.GetXaxis().GetBinCenter(ibin))
        y.append(center)
        up.append(center + width)
        down.append(max(center - width, 0))

    n = len(x)
    grband = ROOT.TGraph(2 * n)
    for i in range(n):
        grband.SetPoint(i, x[i], up[i])
        grband.SetPoint(n + i, x[n - i - 1], down[n - i - 1])

    grband.SetFillStyle(3013)
    grband.SetFillColor(16)
    return grband
