import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

"""This script investigates the bad tracks in the GNN-based tracking results.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from acctrack.utils.utils_plot import add_yline

def read_track_perf_file(trk_perf_fname: str) -> pd.DataFrame:
    column_names = ["idx", "chi2/ndof", 'px', 'py', 'pz',
                    'pt', 'd0', 'z0', 'charge', 'qoverp']
    
    df = pd.read_csv(trk_perf_fname, sep=",",
                    header=None,
                    names=column_names)

    df['phi'] = np.arctan2(df['py'], df['px'])
    df['theta'] = np.arctan2(df['pt'], df['pz'])
    df['eta'] = -np.log(np.tan(df['theta']/2.))

    fiducial_cuts = (df['z0'].abs() < 200) & (df['d0'].abs() < 2) \
        & (df['pt'] >= 1000 & (df['pt'] < 10_000))
    df = df[fiducial_cuts]

    return df


def plot_bad_tracks(cfg: DictConfig) -> None:
    """Plot the bad tracks in the GNN-based tracking results.
    """
    track_perf_path = cfg.track_perf_path
    os.makedirs(cfg.output_dir, exist_ok=True)

    event_files = os.listdir(track_perf_path)
    print("Total number of events: {} will be processed by {} workers".format(
        len(event_files), cfg.num_workers))

    outname = os.path.join(cfg.output_dir, 'bad_tracks.parquet')
    if os.path.exists(outname) and not cfg.force_overwrite:
        print(f"Output file {outname} already exists.")
        df = pd.read_parquet(outname)
    else:
        print(f"Output file {outname} does not exist. Will be created.")    
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            futures = [executor.submit(
                read_track_perf_file, os.path.join(track_perf_path, event_file)) \
                    for event_file in event_files]

            dfs = [future.result() for future in concurrent.futures.as_completed(futures)]
            df = pd.concat(dfs, ignore_index=True)

        df.to_parquet(outname)
    
    x_variables = ['eta', 'phi', 'pt']
    x_labels = [r'$\eta$', r'$\phi$', r'$p_T$']
    threshold = 9
    for xvar,xlabel in zip(x_variables, x_labels):
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        xmin, xmax = df[xvar].min(), df[xvar].max()
        if xvar == 'pt':
            xmin, xmax = 0, 10_000

        ax.hist2d(
            df[xvar], df['chi2/ndof'], 
            bins=(100, 100), range=((xmin, xmax), (0.1, 100)),
            norm=colors.LogNorm(), cmap="Blues"
        )
        add_yline(ax, threshold, (xmin, xmax), color='red', lw=2)
        # sns.histplot(df[ (df['chi2/ndof'] < 100) & (df['chi2/ndof'] > 0.1)],
        #             x=xvar, y='chi2/ndof', 
        #             log_scale=(False,True), ax=ax, cbar=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\chi^2$ / ndof')
        plt.savefig(os.path.join(cfg.output_dir, f'bad_tracks_chi2_vs_{xvar}.png'))
        plt.cla()
    


@hydra.main(config_path=root / "configs", config_name="plot_bad_tracks.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    plot_bad_tracks(cfg)

if __name__ == "__main__":
    main()