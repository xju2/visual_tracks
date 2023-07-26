"""This module provides a class that takes following parameters as inputs
* model_path: path to the trained model
* data_path: path to the data to be used for inference
* output_path: path to the output file
* kwargs: other parameters that are needed for inference
"""
from typing import Optional
from pathlib import Path
from acctrack.io.pyg_data_reader import TrackGraphDataReader
from acctrack.tools.utils_graph import build_edges

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

    def inference(self, evtid: int, knn_backend: Optional[str] = None) -> Tensor:
        data = self.data_reader_training.read(evtid)
        if self.stage_name == "graph_construction":
            node_features = self.config["node_features"]
            node_scales = torch.Tensor(self.config["node_scales"])
            input_data = torch.stack([data[x] for x in node_features], dim=-1).float()
            features = input_data / node_scales
            # print("model is at:", self.model_reader.model.device, "and input data is at:", features.device)
            embedding = self.model_reader.predict(features)

            r_max, k_max = self.config["r_infer"], self.config["knn_infer"]
            edge_index = build_edges(embedding, r_max=r_max, k_max=k_max, backend=knn_backend)

            track_edges = data['track_edges']
            # get *undirected* graph
            track_edges = torch.cat([track_edges, track_edges.flip(0)], dim=-1)
            print("track_edge shape:", track_edges.shape, "edge_index shape:", edge_index.shape)

            # then evaluate the predicted edges with the true edges

            return edge_index
        else:
            pass

