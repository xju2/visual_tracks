from typing import Dict
import ROOT
import random
import acctrack.tools.AtlasStyle   # noqa

from omegaconf import DictConfig

class Canvas:
    def __init__(self,
                 size: Dict,
                 atlas_label: Dict,
                 other_label: Dict,
                 legend: Dict) -> None:
        self.size = size
        self.atlas_label = atlas_label
        self.other_label = other_label
        self.legend = legend

    def create(self, with_ratio: bool) -> None:
        cID = random.randint(1, 10000)
        name = f"canvas{cID}"
        canvas = ROOT.TCanvas(name, name, self.size.width, self.size.height)
        if with_ratio:
            pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1)
            pad1.SetBottomMargin(0)
            pad1.Draw()
            canvas.cd()
            pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.3)
            pad2.SetTopMargin(0)
            pad2.SetBottomMargin(0.5)
            pad2.Draw()
            return canvas, pad1, pad2
        else:
            return canvas, None, None

    def create_legend(self):
        x = self.legend.x
        y = self.legend.y
        width = self.legend.width
        height = self.legend.height
        tsize = self.legend.text_size

        legend = ROOT.TLegend(x, y, x + width, y + height)
        legend.SetTextSize(tsize)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        return legend

    def add_atlas_label(self, with_ratio: bool):
        text = self.atlas_label.text
        x = self.atlas_label.x
        y = self.atlas_label.y
        tsize = self.atlas_label.text_size
        color = self.atlas_label.color

        label = ROOT.TLatex()
        label.SetNDC()
        label.SetTextFont(72)
        label.SetTextColor(color)
        label.SetTextSize(tsize)
        label.DrawLatex(x, y, "ATLAS")
        if text is not None:
            delx = 0.115 * 696 * ROOT.gPad.GetWh() / (472 * ROOT.gPad.GetWw())
            if with_ratio:
                delx = delx * 0.75 * (tsize / 0.05)
            p = ROOT.TLatex()
            p.SetNDC()
            p.SetTextFont(42)
            p.SetTextSize(tsize)
            p.SetTextColor(color)
            p.DrawLatex(x + delx, y, text)

    def add_other_label(self):
        x = self.other_label.x
        y = self.other_label.y
        tsize = self.other_label.text_size
        color = self.other_label.color
        text = self.other_label.text
        if text is not None:
            label = ROOT.TLatex()
            label.SetNDC()
            label.SetTextFont(42)
            label.SetTextColor(color)
            label.SetTextSize(tsize)
            label.DrawLatex(x, y, text)

    def update(self, config: DictConfig):
        if "size" in config:
            self.size.update(config.size)

        if "atlas_label" in config:
            self.atlas_label.update(config.atlas_label)

        if "other_label" in config:
            self.other_label.update(config.other_label)

        if "legend" in config:
            self.legend.update(config.legend)

    def deepupdate(self, config: DictConfig):
        copy = Canvas(self.size, self.atlas_label,
                      self.other_label, self.legend)
        copy.update(config)
        return copy

    def __repr__(self):
        return f"Canvas(size={self.size}, atlas_label={self.atlas_label}, other_label={self.other_label}, legend={self.legend})"

    def __str__(self):
        return f"Canvas(size={self.size}, atlas_label={self.atlas_label}, other_label={self.other_label}, legend={self.legend})"
