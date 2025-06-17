import torch as th
import numpy as np
import scipy.sparse as sp
import dgl
import random
from utils import sparse_mx_to_torch_sparse_tensor

class GraphAugmentation:
    """
    Graph data augmentation module: Provides various data augmentation methods to reduce overfitting in graph neural networks
    """
    @staticmethod
    def random_edge_dropout(graph, dropout_rate=0.1):
        """
        Randomly drop edges
        
        Args:
        graph: DGL graph object
        dropout_rate: Edge dropout ratio (0-1)
        
        Returns:
        augmented_graph: Augmented graph
        """
        if not isinstance(graph, dgl.DGLGraph):
            return graph
            
        # Create a new heterogeneous graph, preserving original structure but only keeping edges we want
        data_dict = {}
        num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
        
        # Determine device of the graph
        device = graph.device
        
        # Process each edge type
        for etype in graph.canonical_etypes:
            src_ntype, rel_type, dst_ntype = etype
            
            # Get number of edges for this type
            num_edges = graph.number_of_edges(etype)
            
            if num_edges == 0:
                # If no edges, add an empty edge list
                data_dict[etype] = (th.tensor([], dtype=th.int64, device=device), 
                                   th.tensor([], dtype=th.int64, device=device))
                continue
            
            # Calculate number of edges to keep
            num_keep = max(1, int(num_edges * (1 - dropout_rate)))  # Keep at least one edge
            
            # Randomly select edges to keep
            perm = th.randperm(num_edges, device=device)
            edges_to_keep = perm[:num_keep]
            
            # Get all source and destination nodes
            src, dst = graph.edges(etype=etype)
            
            # Select edges to keep
            src_keep = src[edges_to_keep]
            dst_keep = dst[edges_to_keep]
            
            # Add kept edges to new data dictionary
            data_dict[etype] = (src_keep, dst_keep)
        
        # Create new heterogeneous graph
        new_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # Copy node features
        for ntype in graph.ntypes:
            for key, feat in graph.nodes[ntype].data.items():
                new_graph.nodes[ntype].data[key] = feat.clone()
        
        # Copy edge features (if any)
        for etype in graph.canonical_etypes:
            if new_graph.number_of_edges(etype) > 0:
                for key, feat in graph.edges[etype].data.items():
                    # Select corresponding edge features using indices
                    src_type, rel_type, dst_type = etype
                    if graph.number_of_edges(etype) == new_graph.number_of_edges(etype):
                        # If edge count is the same, copy directly
                        new_graph.edges[etype].data[key] = feat.clone()
                    else:
                        # Otherwise, need to select corresponding features
                        try:
                            indices = perm[:num_keep]  # Use previously selected edge indices
                            new_graph.edges[etype].data[key] = feat[indices].clone()
                        except Exception as e:
                            print(f"Warning: Could not copy edge feature {key} for edge type {etype}: {e}")
        
        return new_graph

    @staticmethod
    def random_edge_dropout_sparse(sparse_graph, dropout_rate=0.1):
        """
        Randomly drop edges in sparse graph
        
        Args:
        sparse_graph: Graph represented as sparse tensor
        dropout_rate: Edge dropout ratio (0-1)
        
        Returns:
        augmented_graph: Augmented graph
        """
        if not isinstance(sparse_graph, th.Tensor) or not sparse_graph.is_sparse:
            return sparse_graph
            
        # Get indices and values from sparse tensor
        indices = sparse_graph._indices()
        values = sparse_graph._values()
        shape = sparse_graph.shape
        device = sparse_graph.device
        
        # Calculate number of edges to keep
        num_edges = values.size(0)
        num_keep = max(1, int(num_edges * (1 - dropout_rate)))  # Keep at least one edge
        
        # Randomly select edges to keep
        perm = th.randperm(num_edges, device=device)
        keep_indices = perm[:num_keep]
        
        # Create new sparse tensor
        new_indices = indices[:, keep_indices]
        new_values = values[keep_indices]
        
        return th.sparse_coo_tensor(new_indices, new_values, shape, device=device)

    @staticmethod
    def add_random_edges(graph, add_rate=0.05, self_loops=False):
        """
        Randomly add edges
        
        Args:
        graph: DGL graph object
        add_rate: Ratio of new edges to add relative to original edges
        self_loops: Whether to allow self-loops
        
        Returns:
        augmented_graph: Augmented graph
        """
        if not isinstance(graph, dgl.DGLGraph):
            return graph
            
        # Create deep copy of graph
        augmented_graph = graph.clone()
        
        # Determine device of graph
        device = augmented_graph.device
        
        # Process each edge type separately
        for etype in augmented_graph.canonical_etypes:
            src_ntype, rel_type, dst_ntype = etype
            
            # Get number of edges for this type
            num_edges = augmented_graph.number_of_edges(etype)
            if num_edges == 0:
                continue  # Skip edge types with no edges
                
            # Calculate number of edges to add
            num_add = max(1, int(num_edges * add_rate))
                
            # Get number of source and destination nodes
            num_src_nodes = augmented_graph.number_of_nodes(src_ntype)
            num_dst_nodes = augmented_graph.number_of_nodes(dst_ntype)
            
            if num_src_nodes == 0 or num_dst_nodes == 0:
                continue  # Skip node types with no nodes
            
            # Get existing edges to avoid adding duplicates
            existing_edges = set()
            for i, (s, d) in enumerate(zip(*augmented_graph.edges(etype=etype))):
                existing_edges.add((s.item(), d.item()))
            
            # Randomly generate new edges
            new_src = []
            new_dst = []
            attempts = 0
            max_attempts = num_add * 10  # Limit attempts to avoid infinite loop
            
            while len(new_src) < num_add and attempts < max_attempts:
                src_idx = random.randint(0, num_src_nodes - 1)
                dst_idx = random.randint(0, num_dst_nodes - 1)
                
                # Check self-loops if not allowed and source/destination node types are the same
                if not self_loops and src_ntype == dst_ntype and src_idx == dst_idx:
                    attempts += 1
                    continue
                    
                # Check if edge already exists
                if (src_idx, dst_idx) not in existing_edges and (src_idx, dst_idx) not in zip(new_src, new_dst):
                    new_src.append(src_idx)
                    new_dst.append(dst_idx)
                
                attempts += 1
            
            # Add new edges
            if new_src:
                try:
                    augmented_graph.add_edges(
                        th.tensor(new_src, dtype=th.int64, device=device),
                        th.tensor(new_dst, dtype=th.int64, device=device),
                        etype=etype
                    )
                except Exception as e:
                    print(f"Warning: Error adding edges for edge type {etype}: {e}")
        
        return augmented_graph

    @staticmethod
    def feature_noise(features, noise_scale=0.1):
        """
        Add Gaussian noise to features
        
        Args:
        features: Node feature tensor
        noise_scale: Noise standard deviation
        
        Returns:
        noisy_features: Features with added noise
        """
        if features is None:
            return None
            
        if isinstance(features, th.Tensor):
            # Get device of features
            device = features.device
            
            # Generate Gaussian noise with same shape as features on same device
            noise = th.randn_like(features, device=device) * noise_scale
            noisy_features = features + noise
            return noisy_features
        else:
            # If not tensor, try to convert
            try:
                features_tensor = th.tensor(features, dtype=th.float32)
                device = th.device('cuda' if th.cuda.is_available() else 'cpu')
                features_tensor = features_tensor.to(device)
                noise = th.randn_like(features_tensor) * noise_scale
                noisy_features = features_tensor + noise
                return noisy_features
            except Exception as e:
                print(f"Warning: Failed to add noise to features: {e}")
                return features
    
    @staticmethod
    def sparse_graph_noise(graph, noise_scale=0.05):
        """
        Add noise to sparse graph
        
        Args:
        graph: Sparse graph (torch.sparse_coo_tensor)
        noise_scale: Noise standard deviation
        
        Returns:
        noisy_graph: Graph with added noise
        """
        if not isinstance(graph, th.Tensor) or not graph.is_sparse:
            return graph
            
        # Get indices and values from sparse tensor
        indices = graph._indices()
        values = graph._values()
        shape = graph.shape
        device = graph.device
        
        # Add noise to values (ensure on same device)
        noise = th.randn_like(values, device=device) * noise_scale
        noisy_values = values + noise
        
        # Ensure values are in reasonable range (e.g., non-negative)
        noisy_values = th.clamp(noisy_values, min=0.0)
        
        # Create new sparse tensor
        noisy_graph = th.sparse_coo_tensor(indices, noisy_values, shape, device=device)
        return noisy_graph
    
    @staticmethod
    def feature_masking(features, mask_rate=0.1):
        """
        Randomly mask elements in features
        
        Args:
        features: Node feature tensor
        mask_rate: Masking rate (0-1)
        
        Returns:
        masked_features: Masked features
        """
        if features is None:
            return None
            
        if isinstance(features, th.Tensor):
            # Get device of features
            device = features.device
            
            # Create mask tensor on same device
            mask = (th.rand_like(features, device=device) > mask_rate)
            # Apply mask
            masked_features = features * mask
            return masked_features
        else:
            try:
                device = th.device('cuda' if th.cuda.is_available() else 'cpu')
                features_tensor = th.tensor(features, dtype=th.float32, device=device)
                mask = (th.rand_like(features_tensor) > mask_rate)
                masked_features = features_tensor * mask
                return masked_features
            except Exception as e:
                print(f"Warning: Failed to mask features: {e}")
                return features
    
    @staticmethod
    def mix_up_features(features, alpha=0.2):
        """
        Apply mixup augmentation to features
        
        Args:
        features: Feature tensor of shape (N, D), where N is number of nodes, D is feature dimension
        alpha: Parameter for Beta distribution
        
        Returns:
        mixed_features: Features after mixup
        """
        if features is None or not isinstance(features, th.Tensor):
            return features
            
        # Get device of features
        device = features.device
            
        # Randomly permute node order
        indices = th.randperm(features.size(0), device=device)
        shuffled_features = features[indices]
        
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Mix features
        mixed_features = lam * features + (1 - lam) * shuffled_features
        return mixed_features


# Enhanced version of existing knn_graph function
def augmented_knn_graph(disMat, k, dropout_rate=0.1, add_noise=False, noise_scale=0.1):
    """
    Build enhanced k-nearest neighbor graph (with edge dropout and noise)
    
    Args:
    disMat: Distance/similarity matrix
    k: Number of nearest neighbors
    dropout_rate: Ratio of edges to randomly drop
    add_noise: Whether to add noise
    noise_scale: Noise standard deviation
    
    Returns:
    adj: Enhanced adjacency matrix (sparse format)
    """
    from utils import knn_graph
    
    # First build original kNN graph
    adj = knn_graph(disMat, k)
    
    # Add noise (if needed)
    if add_noise:
        # Get non-zero elements
        rows, cols = adj.nonzero()
        values = adj.data
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, len(values))
        values = values + noise
        
        # Ensure values are in reasonable range
        values = np.clip(values, 0.01, 1.0)
        
        # Rebuild sparse matrix
        adj = sp.coo_matrix((values, (rows, cols)), shape=adj.shape)
    
    # Edge dropout
    if dropout_rate > 0:
        # Get non-zero elements
        rows, cols = adj.nonzero()
        values = adj.data
        
        # Randomly select edges to keep
        num_edges = len(values)
        num_keep = max(1, int(num_edges * (1 - dropout_rate)))
        keep_indices = np.random.choice(num_edges, num_keep, replace=False)
        
        # Get edges to keep
        rows_keep = rows[keep_indices]
        cols_keep = cols[keep_indices]
        values_keep = values[keep_indices]
        
        # Rebuild sparse matrix
        adj = sp.coo_matrix((values_keep, (rows_keep, cols_keep)), shape=adj.shape)
    
    # Ensure symmetry and self-loops
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    
    return adj

# Used for data augmentation in training loop
def augment_graph_data(graph_data, aug_methods=None, aug_params=None):
    """
    Augment graph data
    
    Args:
    graph_data: Dictionary containing various graphs and features
    aug_methods: List of augmentation methods to apply
    aug_params: Dictionary of augmentation method parameters
    
    Returns:
    augmented_data: Augmented graph data
    """
    if aug_methods is None:
        aug_methods = ['edge_dropout']
    if aug_params is None:
        aug_params = {'edge_dropout_rate': 0.1}
    
    # Create a copy to avoid modifying original data
    augmented_data = {}
    for key, value in graph_data.items():
        if value is not None:
            # Deep copy tensor and graph objects
            if isinstance(value, th.Tensor):
                augmented_data[key] = value.clone()
            elif isinstance(value, dgl.DGLGraph):
                augmented_data[key] = value.clone()
            else:
                augmented_data[key] = value
        else:
            augmented_data[key] = None
            
    # Apply specified augmentation methods
    for method in aug_methods:
        if method == 'edge_dropout':
            # Apply edge dropout to various graphs
            edge_dropout_rate = aug_params.get('edge_dropout_rate', 0.1)
            
            # Check and process each type of graph
            for graph_key in ['enc_graph', 'dec_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], dgl.DGLGraph):
                        try:
                            augmented_data[graph_key] = GraphAugmentation.random_edge_dropout(
                                augmented_data[graph_key], edge_dropout_rate)
                        except Exception as e:
                            print(f"Warning: Error applying edge_dropout to {graph_key}: {e}")
            
            # Process sparse tensor graphs
            for graph_key in ['drug_graph', 'disease_graph', 
                             'drug_feature_graph', 'disease_feature_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], th.Tensor) and augmented_data[graph_key].is_sparse:
                        try:
                            augmented_data[graph_key] = GraphAugmentation.random_edge_dropout_sparse(
                                augmented_data[graph_key], edge_dropout_rate)
                        except Exception as e:
                            print(f"Warning: Error applying edge_dropout to sparse tensor {graph_key}: {e}")
        
        elif method == 'add_random_edges':
            # Randomly add edges to graphs
            add_edge_rate = aug_params.get('add_edge_rate', 0.05)
            
            for graph_key in ['enc_graph', 'dec_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], dgl.DGLGraph):
                        try:
                            augmented_data[graph_key] = GraphAugmentation.add_random_edges(
                                augmented_data[graph_key], add_edge_rate)
                        except Exception as e:
                            print(f"Warning: Error applying add_random_edges to {graph_key}: {e}")
        
        elif method == 'feature_noise':
            # Add noise to node features
            feature_noise_scale = aug_params.get('feature_noise_scale', 0.1)
            sim_noise_scale = aug_params.get('sim_noise_scale', 0.05)
            
            for feat_key, noise_scale in [
                ('drug_feat', feature_noise_scale),
                ('disease_feat', feature_noise_scale),
                ('drug_sim_feat', sim_noise_scale),
                ('disease_sim_feat', sim_noise_scale)
            ]:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.feature_noise(
                            augmented_data[feat_key], noise_scale)
                    except Exception as e:
                        print(f"Warning: Error applying feature_noise to {feat_key}: {e}")
        
        elif method == 'graph_noise':
            # Add noise to graph structure
            graph_noise_scale = aug_params.get('graph_noise_scale', 0.05)
            
            for graph_key in ['drug_graph', 'disease_graph', 
                             'drug_feature_graph', 'disease_feature_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    # Only process sparse tensors
                    if isinstance(augmented_data[graph_key], th.Tensor) and augmented_data[graph_key].is_sparse:
                        try:
                            augmented_data[graph_key] = GraphAugmentation.sparse_graph_noise(
                                augmented_data[graph_key], graph_noise_scale)
                        except Exception as e:
                            print(f"Warning: Error applying graph_noise to {graph_key}: {e}")
        
        elif method == 'feature_masking':
            # Randomly mask features
            feature_mask_rate = aug_params.get('feature_mask_rate', 0.1)
            
            for feat_key in ['drug_feat', 'disease_feat']:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.feature_masking(
                            augmented_data[feat_key], feature_mask_rate)
                    except Exception as e:
                        print(f"Warning: Error applying feature_masking to {feat_key}: {e}")
        
        elif method == 'mix_up':
            # Apply mixup to node features
            mixup_alpha = aug_params.get('mixup_alpha', 0.2)
            
            for feat_key in ['drug_feat', 'disease_feat']:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.mix_up_features(
                            augmented_data[feat_key], mixup_alpha)
                    except Exception as e:
                        print(f"Warning: Error applying mix_up to {feat_key}: {e}")
    
    return augmented_data