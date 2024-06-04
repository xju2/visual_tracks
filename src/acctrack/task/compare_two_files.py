from pathlib import Path

from acctrack.task.base import TaskBase
from acctrack.utils import get_pylogger
from acctrack.tools.reader import TH1FileHandle
from acctrack.tools.ratio import create_ratio
from acctrack.tools import adder

logger = get_pylogger(__name__)


class CompareTwoIdentidicalFiles(TaskBase):
    def __init__(
        self,
        reference_file: TH1FileHandle,
        comparator_file: TH1FileHandle,
        with_ratio: bool = True,
        outdir: str = ".",
        name: str = "CompareTwoIdentidicalFiles",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["reference_file", "comparator_file"])
        self.ref_file = reference_file
        self.comparator_file = comparator_file
        # self.plotter = Plotter()

    def run(self) -> None:
        print(self.ref_file)
        print(self.comparator_file)
        print(self.histograms)
        print(self.canvas)

        # check if canvas needs adjustment
        if "canvas" in self.histograms.config:
            self.canvas.update(self.histograms.config["canvas"])

        with_ratio = self.hparams.with_ratio
        for histogram in self.histograms:
            hist_ref, hist_ref_copy = self.ref_file.read(histogram)
            hist_comparator, hist_comparator_copy = self.comparator_file.read(histogram)

            # check if canvas needs adjustment for this histogram
            if "canvas" in histogram.hparams:
                canvas_cls = self.canvas.deepupdate(histogram.hparams.canvas)
            else:
                canvas_cls = self.canvas

            if "with_ratio" in histogram.hparams:
                with_ratio_this = histogram.hparams.with_ratio
            else:
                with_ratio_this = with_ratio

            canvas, pad1, pad2 = canvas_cls.create(with_ratio_this)
            canvas.cd()

            histname = Path(histogram.hparams.histname).name
            is_logy = histogram.hparams.is_logy

            hist_ref_copy.SetLineColor(9000)
            hist_ref.SetLineColor(9000)
            hist_ref.SetMarkerSize(0)

            hist_comparator.SetMarkerColor(9001)
            hist_comparator.SetLineColor(9001)
            hist_comparator.SetMarkerStyle(8)
            hist_comparator.SetMarkerSize(0.9)

            if with_ratio_this:
                pad1.cd()
                if is_logy:
                    pad1.SetLogy()
                hist_ref_copy.GetYaxis().SetTitleSize(0.065)
                hist_ref_copy.GetYaxis().SetTitleOffset(0.75)
                hist_ref_copy.GetYaxis().SetLabelSize(0.06)
            else:
                if is_logy:
                    canvas.SetLogy()
                hist_ref_copy.GetYaxis().SetTitleSize(0.06)
                hist_ref_copy.GetYaxis().SetLabelSize(0.055)
                hist_ref_copy.GetXaxis().SetTitleSize(0.06)
                hist_ref_copy.GetXaxis().SetLabelSize(0.055)

            hist_ref_copy.Draw("hist")
            hist_ref.Draw("same EP")
            hist_comparator.Draw("same ep")

            self.canvas.add_atlas_label(with_ratio_this)
            legend = self.canvas.create_legend()
            legend.AddEntry(hist_ref_copy, self.ref_file.hparams.name, "lep")
            legend.AddEntry(hist_comparator, self.comparator_file.hparams.name, "ep")
            legend.Draw()

            self.canvas.add_other_label()
            if with_ratio_this:
                pad2.cd()
                ratio = create_ratio(hist_ref_copy, hist_comparator_copy)
                if histogram.hparams.ratio_ylim is not None:
                    ratio.GetYaxis().SetRangeUser(*histogram.hparams.ratio_ylim)
                if histogram.hparams.ratio_ylabel is not None:
                    ratio.GetYaxis().SetTitle(histogram.hparams.ratio_ylabel)

                ratio.Draw("EP")
                adder.add_line(ratio, 1.0)

            # write the canvas to file
            outname = histname + "-withratio" if with_ratio_this else histname
            if self.canvas.atlas_label.text is not None:
                outname += f"-{self.canvas.atlas_label.text}"
            outname += ".pdf"
            outname = Path(self.hparams.outdir) / outname
            canvas.SaveAs(str(outname))
