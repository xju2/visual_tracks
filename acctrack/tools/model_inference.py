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
from acctrack.tools.edge_perf import EdgePerformance

import yaml
import torch
from torch import Tensor

class ModelLoader:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        print("Loading model from", self.model_path)
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def predict(self, data) -> Tensor:
        return self.model.forward(data)

class TorchModelInference:
    def __init__(self, config_fname: str,
                 data_type: str,
                 model_path: str,
                 output_path: str, name="TorchModelInference") -> None:
        self.name = name
        # data type can be trainset, valset, or testset
        assert data_type in ["trainset", "valset", "testset"], \
            "data_type must be trainset, valset, or testset"

        with open(config_fname, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config

        self.model_reader = ModelLoader(model_path)
        self.model_reader.load()

        data_path = Path(config['input_dir'])
        # self.data_reader_training = TrackGraphDataReader(data_path / "trainset", name="Training")
        # self.data_reader_validation = TrackGraphDataReader(data_path / "valset", name="Validation")
        # self.data_reader_test = TrackGraphDataReader(data_path / "testset", name="Test")

        self.data_reader = TrackGraphDataReader(data_path / data_type, name=data_type)

        self.output_path = output_path
        self.stage_name = config['stage']

        self.edge_perf = EdgePerformance(self.data_reader)

    def inference(self, evtid: int, radius: float = 0.1, knn: int = 1000,
                  knn_backend: Optional[str] = None) -> Tensor:

        self.data_reader.read(evtid)  # tell the reader to read the event
        if self.stage_name == "graph_construction":
            node_features = self.config["node_features"]
            node_scales = torch.Tensor(self.config["node_scales"])

            features = self.data_reader.get_node_features(node_features, node_scales)
            # print("model is at:", self.model_reader.model.device, "and input data is at:", features.device)
            embedding = self.model_reader.predict(features)

            edge_index = build_edges(embedding, r_max=radius, k_max=knn, backend=knn_backend)
            truth_labels, true_edges, per_edge_efficiency, per_edge_purity = self.edge_perf.eval(edge_index)

            return edge_index, truth_labels, true_edges
        else:
            pass

