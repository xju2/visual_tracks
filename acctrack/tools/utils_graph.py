import logging
from typing import Optional, Tuple, Union


import torch.nn as nn
import torch
from torch_geometric.nn import radius

try:
    import frnn
    FRNN_AVAILABLE = True
    logging.warning("FRNN is available")
except ImportError:
    FRNN_AVAILABLE = False
    logging.warning("FRNN is not available, install it at https://github.com/murnanedaniel/FRNN. Using PyG radius instead.")

faiss_avail = False
try:
    import faiss
    faiss_avail = True
    logging.warning("FAISS is available")
except ImportError:
    faiss_avail = False
    logging.warning("FAISS is not available, install it at \"conda install faiss-gpu -c pytorch\" or \
                    \"pip install faiss-gpu\". Using PyG radius instead.")

if not torch.cuda.is_available():
    FRNN_AVAILABLE = False
    logging.warning("FRNN is not available, as no GPU is available")


# ---------------------------- Dataset Processing -------------------------

# ---------------------------- Edge Building ------------------------------

def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    r_max: float = 1.0,
    k_max: int = 10,
    return_indices: bool = False,
    backend: str = "FRNN"
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

    # Type hint
    if backend == "FRNN" and FRNN_AVAILABLE:
        # Compute edges
        dists, idxs, _, _ = frnn.frnn_grid_points(
            points1=query.unsqueeze(0),
            points2=database.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )

        idxs: torch.Tensor = idxs.squeeze().int()
        ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
        positive_idxs = idxs >= 0
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    elif faiss_avail and "FAISS" in backend:
        if backend == "FAISS-CPU":
            dists, idxs = faiss.knn(query.detach().cpu().numpy(), database.detach().cpu().numpy(), k_max)
        else:
            dists, idxs = faiss.knn_gpu(query, database, k_max)
        print(dist.shape, idxs.shape)
        dists, idxs = dists[:, 1:], idxs[:, 1:]  # Remove self-loops
        ind = torch.arange(idxs.shape[0], device=query.device).repeat(idxs.shape[1], 1).T.int()
        positive_idxs = (dists <= r_max**2)
        edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).int()
    else:
        edge_list = radius(database, query, r=r_max, max_num_neighbors=k_max)

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return (edge_list, dists, idxs, ind) if (return_indices and backend=="FRNN") else edge_list


def graph_intersection(input_pred_graph, input_truth_graph, return_y_pred=True, return_y_truth=False, return_pred_to_truth=False, return_truth_to_pred=False, unique_pred=True, unique_truth=True):
    """
    An updated version of the graph intersection function, which is around 25x faster than the
    Scipy implementation (on GPU). Takes a prediction graph and a truth graph, assumed to have unique entries.
    If unique_pred or unique_truth is False, the function will first find the unique entries in the input graphs, and return the updated edge lists.
    """

    if not unique_pred:
        input_pred_graph = torch.unique(input_pred_graph, dim=1)
    if not unique_truth:
        input_truth_graph = torch.unique(input_truth_graph, dim=1)

    unique_edges, inverse = torch.unique(torch.cat([input_pred_graph, input_truth_graph], dim=1), dim=1, sorted=False, return_inverse=True, return_counts=False)

    inverse_pred_map = torch.ones_like(unique_edges[1]) * -1
    inverse_pred_map[inverse[:input_pred_graph.shape[1]]] = torch.arange(input_pred_graph.shape[1], device=input_pred_graph.device)

    inverse_truth_map = torch.ones_like(unique_edges[1]) * -1
    inverse_truth_map[inverse[input_pred_graph.shape[1]:]] = torch.arange(input_truth_graph.shape[1], device=input_truth_graph.device)

    pred_to_truth = inverse_truth_map[inverse][:input_pred_graph.shape[1]]
    truth_to_pred = inverse_pred_map[inverse][input_pred_graph.shape[1]:]

    return_tensors = []

    if not unique_pred:
        return_tensors.append(input_pred_graph)
    if not unique_truth:
        return_tensors.append(input_truth_graph)
    if return_y_pred:
        y_pred = pred_to_truth >= 0
        return_tensors.append(y_pred)
    if return_y_truth:
        y_truth = truth_to_pred >= 0
        return_tensors.append(y_truth)
    if return_pred_to_truth:
        return_tensors.append(pred_to_truth)
    if return_truth_to_pred:
        return_tensors.append(truth_to_pred)

    return return_tensors if len(return_tensors) > 1 else return_tensors[0]