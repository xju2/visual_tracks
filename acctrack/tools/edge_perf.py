"""This module evaluates the per-edge performance"""

from typing import Any, Optional
import torch
from acctrack.tools.utils_graph import graph_intersection
from acctrack.viewer.classification import plot_metrics
from acctrack.io.pyg_data_reader import TrackGraphDataReader

class EdgePerformance:
    def __init__(self, reader: TrackGraphDataReader, name="EdgePerformance"):
        self.reader = reader
        self.name = name

    def eval(self, edge_index: torch.Tensor,
             edge_socre: Optional[torch.Tensor] = None,
             edge_weights: Optional[torch.Tensor] = None,
             edge_weight_cuts: float = 0,
             outname: Optional[str] = None,
             *args: Any, **kwds: Any) -> Any:
        """Evaluate the per-edge performance"""

        true_edges = self.reader.data['track_edges']
        # get *undirected* graph
        true_edges = torch.cat([true_edges, true_edges.flip(0)], dim=-1)

        num_true_edges = true_edges.shape[1]

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

        # look at only the edges from nodes of interests.
        masks = self.reader.get_edge_masks()
        # undirected graph, so double the masks
        masks = torch.cat([masks, masks], dim=-1)
        # use the sender of the edge to quanlify the edge

        masked_true_edges = true_edges[:, masks]
        num_masked_true_edges = masked_true_edges.shape[1]
        masked_truth_labels = graph_intersection(edge_index, masked_true_edges)
        num_masked_true_reco_edges = masked_truth_labels.sum().item()
        per_masked_edge_efficiency = 100. * num_masked_true_reco_edges / num_masked_true_edges
        frac_masked_true_reco_edges = 100. * num_masked_true_edges / num_true_edges
        print("Only {:.3f}% of true edges are of interests (signal)".format(frac_masked_true_reco_edges))
        print("True Reco Signal Edges {:,}, True Signal Edges {:,}, Per-edge signal efficiency: {:.3f}%".format(
            num_masked_true_reco_edges, num_masked_true_edges, per_masked_edge_efficiency))

        # use the edge score to evaluate the edge classifier performance
        if edge_socre is not None:
            plot_metrics(edge_socre, truth_labels, outname=outname)
            if edge_weights is not None:
                target_score, target_truth = edge_socre[edge_weights > edge_weight_cuts], truth_labels[edge_weights > edge_weight_cuts]
                plot_metrics(target_score, target_truth, outname=outname + "-target")

        return truth_labels, true_edges, per_edge_efficiency, per_edge_purity
