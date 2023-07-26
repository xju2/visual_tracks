"""This module evaluates the per-edge performance"""

from typing import Any, Optional
import torch
from acctrack.tools.utils_graph import graph_intersection
from acctrack.viewer.classification import plot_metrics


class EdgePerformance:
    def __init__(self, name="EdgePerformance"):
        self.name = name

    def eval(self, edge_index: torch.Tensor, true_edges: torch.Tensor,
             edge_socre: Optional[torch.Tensor] = None,
             edge_weights: Optional[torch.Tensor] = None,
             edge_weight_cuts: float = 0,
             outname: Optional[str] = None,
             *args: Any, **kwds: Any) -> Any:
        """Evaluate the per-edge performance"""

        truth_labels = graph_intersection(edge_index, true_edges)

        # per-edge efficiency
        num_true_reco_edges = truth_labels.sum().item()
        per_edge_efficiency = 100. * num_true_reco_edges / num_true_edges
        print("True Reco Edges {:,}, True Edges {:,}, Per-edge efficiency: {:.3f}%".format(
            num_true_reco_edges, num_true_edges, per_edge_efficiency))

        # per-edge purity
        num_true_edges, num_reco_edges = true_edges.shape[1], edge_index.shape[1]
        per_edge_purity = 100. * num_true_edges / num_reco_edges
        print("True Edges {:,}, Reco Edges {:,}, Per-edge purity: {:.3f}%".format(
            num_true_edges, num_reco_edges, per_edge_purity))

        # use the edge score to evaluate the edge classifier performance
        if edge_socre is not None:
            plot_metrics(edge_socre, truth_labels, outname=outname)
            if edge_weights is not None:
                target_score, target_truth = edge_socre[edge_weights > edge_weight_cuts], truth_labels[edge_weights > edge_weight_cuts]
                plot_metrics(target_score, target_truth, outname=outname + "-target")

        return truth_labels, per_edge_efficiency, per_edge_purity
