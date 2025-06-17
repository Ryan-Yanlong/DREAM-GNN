import csv
import random
import torch as th
import numpy as np
import torch.nn as nn

from scipy import sparse as sp
from collections import OrderedDict


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    # Return sparse tensor on CPU, will be moved to GPU later when needed
    return th.sparse_coo_tensor(indices, values, shape)


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'elu':
            return nn.ELU()
        elif act == 'selu':
            return nn.SELU()
        else:
            raise NotImplementedError
    else:
        return act


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def common_loss(emb1, emb2):
    emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
    emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = th.matmul(emb1, emb1.t())
    cov2 = th.matmul(emb2, emb2.t())
    cost = th.mean((cov1 - cov2) ** 2)
    return cost


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


def knn_graph(disMat, k):
    # Ensure k doesn't exceed distance matrix columns
    n = disMat.shape[0]
    k_actual = min(k, n - 1)  # Subtract 1 to avoid self-loops
   
    if k_actual <= 0:
        # If k is too small, only return self-loops
        return sp.eye(n, dtype=np.float32)
    
    try:
        # Use argpartition to find k nearest neighbors
        k_neighbor = np.argpartition(-disMat, kth=k_actual, axis=1)[:, :k_actual]
        
        # Create row and column indices
        row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
        col_index = k_neighbor.reshape(-1)
        
        # Ensure indices don't exceed bounds
        valid_mask = (col_index >= 0) & (col_index < n)
        row_index = row_index[valid_mask]
        col_index = col_index[valid_mask]
        
        # Create sparse matrix
        edges = np.array([row_index, col_index]).astype(int).T
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(n, n), dtype=np.float32)
        
        # Build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        return adj
    except Exception as e:
        print(f"Error in knn_graph: {e}")
        # Return identity matrix as fallback
        return sp.eye(n, dtype=np.float32)