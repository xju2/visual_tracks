"""This module provides a class that takes following parameters as inputs
* model_path: path to the trained model
* data_path: path to the data to be used for inference
* output_path: path to the output file
* kwargs: other parameters that are needed for inference
"""
from typing import Optional
from pathlib import Path
from acctrack.io.pyg_data_reader import TrackGraphDataReader
from acctrack.tools.utils_graph import build_edges, graph_intersection

import yaml
import torch
from torch import Tensor

class ModelLoader:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def predict(self, data) -> Tensor:
        return self.model.forward(data)

class TorchModelInference:
    def __init__(self, config_fname: str, model_path: str,
                 output_path: str, name="TorchModelInference") -> None:

        with open(config_fname, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config

        self.model_reader = ModelLoader(model_path)
        self.model_reader.load()

        data_path = Path(config['input_dir'])
        self.data_reader_training = TrackGraphDataReader(data_path / "trainset", name="Training")
        self.data_reader_validation = TrackGraphDataReader(data_path / "valset", name="Validation")
        self.data_reader_test = TrackGraphDataReader(data_path / "testset", name="Test")

        self.output_path = output_path
        self.stage_name = config['stage']

    def inference(self, evtid: int, radius: float = 0.1, knn: int = 1000,
                  knn_backend: Optional[str] = None) -> Tensor:
        data = self.data_reader_training.read(evtid)
        if self.stage_name == "graph_construction":
            node_features = self.config["node_features"]
            node_scales = torch.Tensor(self.config["node_scales"])
            input_data = torch.stack([data[x] for x in node_features], dim=-1).float()
            features = input_data / node_scales
            # print("model is at:", self.model_reader.model.device, "and input data is at:", features.device)
            embedding = self.model_reader.predict(features)

            edge_index = build_edges(embedding, r_max=radius, k_max=knn, backend=knn_backend)

            track_edges = data['track_edges']
            # get *undirected* graph
            track_edges = torch.cat([track_edges, track_edges.flip(0)], dim=-1)

            # per-edge purity
            num_true_edges, num_reco_edges = track_edges.shape[1], edge_index.shape[1]
            per_edge_purity = 100. * num_true_edges / num_reco_edges
            print("True Edges {:,}, Reco Edges {:,}, Per-edge purity: {:.4f}%".format(
                num_true_edges, num_reco_edges, per_edge_purity))

            # per-edge efficiency
            truth_labels = graph_intersection(edge_index, track_edges)
            num_true_reco_edges = truth_labels.sum().item()
            per_edge_efficiency = 100. * num_true_reco_edges / num_reco_edges
            print("True Reco Edges {:,}, Reco Edges {:,}, Per-edge efficiency: {:.4f}%".format(
                num_true_reco_edges, num_reco_edges, per_edge_efficiency))

            return edge_index, truth_labels
        else:
            pass

