from typing import Tuple, Optional

from pathlib import Path
from acctrack.hparams_mixin import HyperparametersMixin
from omegaconf import DictConfig, OmegaConf

from acctrack.utils import get_pylogger

log = get_pylogger(__name__)

class HistogramOptions(HyperparametersMixin):
    def __init__(self,
                 histname: str,
                 xlabel: Optional[str] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylabel: Optional[str] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 is_data: bool = False,
                 is_logy: bool = False,
                 rebin: Optional[int] = None,
                 ratio_ylim: Optional[Tuple[float, float]] = None,
                 ratio_ylabel: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

    def __eq__(self, other) -> bool:
        return self.hparams.histname == other.hparams.histname

    def __gt__(self, other) -> bool:
        return self.hparams.histname >= other.hparams.histname

    def __str__(self) -> str:
        return "HistogramOptions:\n" + super().__str__()

    def __repr__(self) -> str:
        return str(self)

class Histograms:
    def __init__(self, config: DictConfig) -> None:
        self._histograms = []
        self.config = dict([(key, value)
                            for key, value in config.items()
                            if key != "histograms"])
        self.use_all = self.config.pop("use_all_histograms", False)
        if self.use_all:
            log.info("Using all histograms in the file")

        self.parse_config(config)

    def parse_config(self, config: DictConfig) -> None:
        """Parse the config"""

        if "histograms" not in config:
            if self.use_all:
                return
            else:
                raise ValueError("No histograms found in the config!")

        def create_histogram(in_config):
            in_cfg = OmegaConf.merge(self.config, in_config)
            if "histo_dir" in config:
                in_cfg.histname = str(Path(config.histo_dir, in_cfg.histname))
            out_histo = HistogramOptions(**in_cfg)
            self._histograms.append(out_histo)

        for hist_cfg in config.histograms:
            histname = hist_cfg.histname

            if isinstance(histname, list):
                # multiple histograms per config
                # check all histograms have the same length
                if not all([len(histname) == len(value)
                            for key, value in hist_cfg.items()]):
                    # find the key with the wrong length
                    for key, value in hist_cfg.items():
                        if len(value) != len(histname):
                            raise ValueError(f"Length of `{key}` "
                                             "is not the same as histnames "
                                             f"{histname[0]}, {len(histname)}")

                for idx in range(len(histname)):
                    cfg = OmegaConf.create(dict([(key, value[idx])
                                                 for key, value in hist_cfg.items()]))
                    create_histogram(cfg)

            elif isinstance(histname, str):
                # single histogram per config
                create_histogram(hist_cfg)

    def __iadd__(self, other):
        self._histograms.extend(other._histograms)
        return self

    def __iter__(self):
        return iter(self._histograms)

    def __len__(self):
        return len(self._histograms)

    def __getitem__(self, index):
        return self._histograms[index]

    def __str__(self) -> str:
        outstr = "Total histograms: {:,}\n".format(len(self._histograms))
        for hist in self._histograms:
            outstr += "\t" + hist.hparams.histname + "\n"
        return outstr

    def get_all_histograms(self, file_handle):
        """Get all histograms from the file."""
        histogram_names = file_handle.get_all_histogram_names()
        all_histograms = [HistogramOptions(histname=histname)
                          for histname in histogram_names]
        if self._histograms:
            # merge the existing histogram configurations with the
            # default ones
            self.merge(all_histograms)
        else:
            self._histograms = all_histograms
        log.info("Read {:,} histograms from the file".format(len(self._histograms)))

    def merge(self, other):
        """Merge the histograms in `other` with the existing ones."""
        for hist in other:
            if hist not in self._histograms:
                self._histograms.append(hist)
                log.info("Added histogram: {}".format(hist.hparams.histname))
