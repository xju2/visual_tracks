"""This module evaluates the per-edge performance"""

from typing import Any, Optional, Union

import numpy as np
import torch
from acctrack.tools.utils_graph import graph_intersection
from acctrack.viewer.classification import plot_metrics
from acctrack.io.pyg_data_reader import TrackGraphDataReader


class EdgePerformance:
    def __init__(self, reader: TrackGraphDataReader, name="EdgePerformance"):
        self.reader = reader
        self.name = name

    def eval(self, edge_index: torch.Tensor, *args: Any, **kwds: Any) -> Any:
        """Evaluate the per-edge performance"""

        true_edges = self.reader.data["track_edges"]
        # get *undirected* graph
        true_edges = torch.cat([true_edges, true_edges.flip(0)], dim=-1)

        num_true_edges = true_edges.shape[1]

        truth_labels = graph_intersection(edge_index, true_edges)

        # per-edge efficiency
        num_true_reco_edges = truth_labels.sum().item()
        per_edge_efficiency = 100.0 * num_true_reco_edges / num_true_edges
        print(
            "True Reco Edges {:,}, True Edges {:,}, Per-edge efficiency: {:.3f}%".format(
                num_true_reco_edges, num_true_edges, per_edge_efficiency
            )
        )

        # per-edge purity
        num_true_edges, num_reco_edges = true_edges.shape[1], edge_index.shape[1]
        per_edge_purity = 100.0 * num_true_edges / num_reco_edges
        print(
            "True Edges {:,}, Reco Edges {:,}, Per-edge purity: {:.3f}%".format(
                num_true_edges, num_reco_edges, per_edge_purity
            )
        )

        # look at only the edges from nodes of interests.
        masks = self.reader.get_edge_masks()
        # undirected graph, so double the masks
        masks = torch.cat([masks, masks], dim=-1)
        # use the sender of the edge to quanlify the edge

        masked_true_edges = true_edges[:, masks]
        num_masked_true_edges = masked_true_edges.shape[1]
        masked_truth_labels = graph_intersection(edge_index, masked_true_edges)
        num_masked_true_reco_edges = masked_truth_labels.sum().item()
        per_masked_edge_efficiency = (
            100.0 * num_masked_true_reco_edges / num_masked_true_edges
        )
        frac_masked_true_reco_edges = 100.0 * num_masked_true_edges / num_true_edges
        print(
            "Only {:.3f}% of true edges are of interests (signal)".format(
                frac_masked_true_reco_edges
            )
        )
        print(
            "True Reco Signal Edges {:,}, True Signal Edges {:,}, Per-edge signal efficiency: {:.3f}%".format(
                num_masked_true_reco_edges,
                num_masked_true_edges,
                per_masked_edge_efficiency,
            )
        )

        return truth_labels, true_edges, per_edge_efficiency, per_edge_purity

    def eval_edge_scores(
        self,
        edge_score: Union[torch.Tensor, np.ndarray],
        truth_labels: Union[torch.Tensor, np.ndarray],
        edge_weights: Optional[torch.Tensor] = None,
        edge_weight_cuts: float = 0,
        outname: Optional[str] = None,
    ):
        """Evaluate the per-edge performance given the edge scores.
        If edge_weights is not None, only plot the edges with weights > edge_weight_cuts.
        Edge weights are used mostly to remove edges that are true but not of interests (non-signal edges).
        """
        if isinstance(edge_score, torch.Tensor):
            edge_score = edge_score.detach().cpu().numpy()
        if isinstance(truth_labels, torch.Tensor):
            truth_labels = truth_labels.detach().cpu().numpy()

        results = plot_metrics(edge_score, truth_labels, outname=outname)
        if edge_weights is not None:
            target_score, target_truth = (
                edge_score[edge_weights > edge_weight_cuts],
                truth_labels[edge_weights > edge_weight_cuts],
            )
            target_results = plot_metrics(
                target_score, target_truth, outname=outname + "-target"
            )
        return results
