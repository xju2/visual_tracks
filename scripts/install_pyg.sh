#!/bin/bash

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install  --no-cache-dir --force-reinstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch_geometric==2.6.1
