#!/usr/bin/env python
"""Scan the knn parameters in graph construction.

This script calculates the efficiency and purity of the knn graph construction
for targeted particles.
"""


from pathlib import Path

import numpy as np
import torch
import tqdm

from acorn.stages.graph_construction.models.metric_learning import (
    GraphDataset,
    MetricLearning,
)
from acorn.stages.graph_construction.models.utils import build_edges, graph_intersection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_one_event(model, event, knn_num, knn_radius: float = 0.1):
    """Process one event and return the true positive edges, total true edges, and total predicted edges."""

    target_tracks = {
        "pt": [1000.0, np.inf],
        "nhits": [3.0, np.inf],
        "primary": True,
        "pdgId": ["not_in", [11, -11]],
    }
    model.apply_target_conditions(event, target_tracks)

    input_data = (
        torch.stack([event[feature] for feature in model.hparams.node_features], dim=1)
        .float()
        .to(device)
    )
    with torch.no_grad():
        embedding = model(input_data)

    pred_edges = build_edges(
        query=embedding,
        database=embedding,
        indices=None,
        r_max=knn_radius,
        k_max=knn_num,
        backend="FRNN",
    )

    true_edges = event.track_edges.to(device)
    pred_edges, _, truth_map = graph_intersection(
        pred_edges,
        true_edges,
        return_y_pred=True,
        return_truth_to_pred=True,
        unique_pred=False,
    )

    true_pos = (truth_map[event.target_mask] >= 0).sum().item()
    tot_truth = truth_map[event.target_mask].shape[0]
    tot_pred = pred_edges.shape[1]

    return true_pos, tot_truth, tot_pred


def scan_one_knn(model, dataset, n_neighbor, radius, num_evts):
    """Scan the knn parameters in graph construction.

    Returns:
        efficiency (float): The efficiency of the graph construction.
        purity (float): The purity of the graph construction.
        num_predicted_edges (int): The number of predicted edges.
    """
    results = []
    for idx in tqdm.tqdm(range(num_evts)):
        results.append(process_one_event(model, dataset[idx], n_neighbor, radius))

    results = np.array(results)
    sum_results = results.sum(axis=0)
    efficiency = sum_results[0] / sum_results[1]
    purity = sum_results[0] / sum_results[2]
    return efficiency, purity, results[2].mean()


class KnnScanner:
    def __init__(self, ckpt_path: str, num_evts: int = 10):
        print(f"Loading the model from {ckpt_path}")
        self.model = MetricLearning.load_from_checkpoint(ckpt_path)
        self.model.eval()

        config = self.model.hparams

        # Load the configuration file.
        self.dataset = GraphDataset(
            config.stage_dir,
            "valset",
            num_events=num_evts,
            stage="predict",
            hparams=config,
        )

        if num_evts == -1:
            num_evts = len(self.dataset)
        else:
            num_evts = min(num_evts, len(self.dataset))
        self.num_evts = num_evts

        ckpt_path = Path(ckpt_path)
        self.postfix = f"evts={num_evts}-{ckpt_path.stem}"
        self.output_dir = ckpt_path.parent

    def scan_radius(
        self,
        start_r: float = 0.1,
        end_r: float = 0.2,
        step: float = 0.01,
        num_neighbors: int = 500,
    ) -> np.ndarray:
        scan_result_file = (
            self.output_dir / f"scan_radius-{num_neighbors}-{self.postfix}.npy"
        )
        if Path(scan_result_file).exists():
            print("Loading the scan result file from ", scan_result_file)
            return np.load(scan_result_file)

        results = []
        for radius in np.arange(start_r, end_r, step):
            efficiency, purity, num_edges = scan_one_knn(
                self.model, self.dataset, num_neighbors, radius, self.num_evts
            )
            results.append((radius, num_neighbors, efficiency, purity, num_edges))
            if num_edges > 20e6:
                break

        results = np.array(results)
        np.save(scan_result_file, results)
        return results

    def scan_neighbors(
        self, start_n: int = 50, end_n: int = 1000, step: int = 50, radius: float = 0.1
    ) -> np.ndarray:
        scan_result_file = (
            self.output_dir / f"scan_neighbors-{radius}-{self.postfix}.npy"
        )
        if Path(scan_result_file).exists():
            print("Loading the scan result file from ", scan_result_file)
            return np.load(scan_result_file)

        num_neighbors = start_n
        results = []
        for num_neighbors in range(start_n, end_n, step):
            efficiency, purity, num_edges = scan_one_knn(
                self.model, self.dataset, num_neighbors, radius, self.num_evts
            )
            results.append((radius, num_neighbors, efficiency, purity, num_edges))
            if num_edges > 20e6:
                break

        results = np.array(results)
        np.save(scan_result_file, results)
        return np.array(results)

    def find_target_radius(
        self, target_efficiency: float, num_neighbors: int = 500
    ) -> float:
        scan_results = self.scan_radius(0.1, 0.5, 0.01, num_neighbors)
        idx = np.argmin(np.abs(scan_results[:, 2] - target_efficiency))
        return scan_results[idx, 0], scan_results[idx, 2]

    def find_max_neighbors(self, radius: float = 0.1) -> int:
        scan_results = self.scan_neighbors(50, 1000, 50, radius)
        idx = np.argmax(scan_results[:, 4])
        return scan_results[idx, 1], scan_results[idx, 4]


def main(ckpt_path: str, num_evts: int = 10, target_eff: float = 0.995):
    """Scan the KNN parameters to find the optimal radius and number of neighbors."""

    scanner = KnnScanner(ckpt_path, num_evts)

    radius, efficiency = scanner.find_target_radius(target_eff, num_neighbors=1000)
    max_neighbor, num_edges = scanner.find_max_neighbors(radius=radius)
    print(
        f"Optimal radius: {radius}, Max neighbors: {max_neighbor}, Efficiency: {efficiency}, Num edges: {num_edges:,}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
