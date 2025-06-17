#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated data loader module.

This module loads drug-disease association data, constructs features, splits cross-validation data,
and builds various graphs based on similarity matrices (e.g., kNN graphs, encoder graphs, decoder graphs).
Experimental setup: Given all known drugs and diseases, predict new drug-disease associations (transductive link prediction).

Main features:
1. Uses the same data splitting method as dataloader.py to ensure consistency
2. Uses mask mechanism to distinguish between training and test sets
3. Does not downsample negative samples, uses all possible drug-disease pairs
4. Builds specific graph structures for each cross-validation fold to ensure test edge information doesn't leak
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.model_selection import KFold

from utils import normalize, sparse_mx_to_torch_sparse_tensor
from augmentation import augmented_knn_graph, GraphAugmentation

# Data file path configuration
_paths = {
    'Gdataset': './raw_data/drug_data/Gdataset/Gdataset.mat',
    'Cdataset': './raw_data/drug_data/Cdataset/Cdataset.mat',
    'Ldataset': './raw_data/drug_data/Ldataset/lagcn',
    'lrssl': './raw_data/drug_data/lrssl/lrssl.mat',
}


class DrugDataLoader(object):
    """
    Drug data loader using consistent splitting method with dataloader.py.
    
    Main functionalities:
      - Load drug-disease association matrix, similarity matrices and pretrained embeddings from .mat files
      - Apply KFold splitting on positive and negative samples separately to create cross-validation data
      - Construct drug and disease feature graphs that don't contain test edge information
      - Build encoder and decoder graphs based on training set association data
      
    Experimental setting: Transductive link prediction, all nodes participate in graph construction,
    but ensuring test edge information doesn't leak into the training process.
    """
    def __init__(self, name, device, symm=True, k=5, use_augmentation=False, aug_params=None):
        """
        Initialize data loader

        Args:
          name: Dataset name (e.g., 'Gdataset' or 'Cdataset')
          device: torch device (e.g., th.device("cpu") or th.device("cuda"))
          symm: Whether to use symmetric normalization
          k: Number of neighbors in kNN graph
          use_augmentation: Whether to enable data augmentation
          aug_params: Data augmentation parameters dictionary
        """
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        self.use_augmentation = use_augmentation
        self.aug_params = aug_params or {}
        self.drug_feature_graph = None
        self.disease_feature_graph = None

        print(f"Starting processing dataset '{self._name}' ...")
        self._dir = os.path.join(_paths[self._name])
        
        # Load raw data
        self._load_raw_data(self._dir, self._name)
        
        # Create cross-validation data dictionary
        self.cv_data_dict = self._create_cv_splits()
        
        # Set to use pretrained embeddings
        self.embedding_mode = "pretrained"
        
        # Generate drug and disease features (embeddings)
        self._generate_feat()
        
        # Construct CV-fold-specific graph structures
        self.cv_specific_graphs = {}
        self._generate_cv_specific_graphs()
        
        # Build training and testing graphs for each fold
        self.data_cv = self._build_all_cv_data()
        
        print(f"[Init] Data loader initialization complete.")

    def _load_raw_data(self, file_path, data_name):
        """
        Load raw data files
        
        Args:
          file_path: Data file path
          data_name: Dataset name
        """
        print(f"[Load Data] Reading data from: {file_path}")
        
        if data_name in ['Gdataset', 'Cdataset','lrssl']:
            data = sio.loadmat(file_path)
            # Transpose to ensure association matrix shape is (num_drug, num_disease)
            self.association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
            # drug_ids format is shape (N, 1), each element e.g. [['DB00014']]
            self.drug_ids = [str(x[0][0]).strip() for x in data['Wrname']] if 'Wrname' in data else None
            
            # Load pretrained embeddings
            if 'drug_embed' in data:
                self.drug_embed = data['drug_embed']
            else:
                print("[Warning] drug_embed not found in dataset. Using random initialization.")
                self.drug_embed = np.random.normal(0, 0.1, (self.association_matrix.shape[0], 768))
                
            if 'disease_embed' in data:
                self.disease_embed = data['disease_embed']
            else:
                print("[Warning] disease_embed not found in dataset. Using random initialization.")
                self.disease_embed = np.random.normal(0, 0.1, (self.association_matrix.shape[1], 768))

        self._num_drug = self.association_matrix.shape[0]
        self._num_disease = self.association_matrix.shape[1]
        print(f"[Load Data] Association matrix shape: {self.association_matrix.shape}")
        print(f"[Load Data] Number of drugs: {self._num_drug}, Number of diseases: {self._num_disease}")

    def _create_cv_splits(self):
        """
        Create cross-validation data splits using the same method as dataloader.py
        
        Returns:
          cv_data: Dictionary, each fold contains [train_data, test_data, unique_values]
        """
        interactions = self.association_matrix
        
        # Get indices of positive and negative samples
        pos_row, pos_col = np.nonzero(interactions)
        neg_row, neg_col = np.nonzero(1 - interactions)
        
        # No downsampling, use all positive and negative samples
        print(f"[CV Split] Positive samples: {len(pos_row)}, Negative samples: {len(neg_row)}")
        
        # Create cross-validation splits
        cv_data = {}
        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        
        for cv_num, ((train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx)) in enumerate(
                zip(kfold.split(pos_row), kfold.split(neg_row))):
            
            # Create training and testing masks
            train_mask = np.zeros_like(interactions, dtype=bool)
            test_mask = np.zeros_like(interactions, dtype=bool)
            
            # Build training and testing edges
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            
            # Merge positive and negative sample edges
            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            
            # Set masks
            train_mask[train_edge[0], train_edge[1]] = True
            test_mask[test_edge[0], test_edge[1]] = True
            
            # Build training data
            train_values = np.zeros(train_edge.shape[1])
            train_values[:len(train_pos_idx)] = 1  # Mark positive samples as 1
            
            # Build testing data
            test_values = np.zeros(test_edge.shape[1])
            test_values[:len(test_pos_idx)] = 1  # Mark positive samples as 1
            
            # Use DataFrame for easier processing
            train_data_info = pd.DataFrame({
                'drug_id': train_edge[0],
                'disease_id': train_edge[1],
                'values': train_values
            })
            
            test_data_info = pd.DataFrame({
                'drug_id': test_edge[0],
                'disease_id': test_edge[1],
                'values': test_values
            })
            
            unique_values = np.array([0, 1])  # Binary classification task
            cv_data[cv_num] = [train_data_info, test_data_info, unique_values]
            
            print(f"[CV Split] Fold {cv_num}: Train = {train_data_info.shape[0]} samples, Test = {test_data_info.shape[0]} samples")
            
        return cv_data

    def _generate_feat(self):
        """
        Construct drug and disease features using pretrained embeddings or random initialization
        """
        if getattr(self, "embedding_mode", "pretrained") == "pretrained":
            if not hasattr(self, 'drug_embed') or not hasattr(self, 'disease_embed'):
                raise ValueError("Pretrained embeddings missing.")
            self.drug_feature = th.FloatTensor(self.drug_embed).to(self._device)
            self.disease_feature = th.FloatTensor(self.disease_embed).to(self._device)
        else:
            print("[Feature] Using randomly initialized embeddings")
            embed_dim = 768
            self.drug_feature = th.FloatTensor(np.random.normal(0, 0.1, (self._num_drug, embed_dim))).to(self._device)
            self.disease_feature = th.FloatTensor(np.random.normal(0, 0.1, (self._num_disease, embed_dim))).to(self._device)
        
        # Normalize features
        self.drug_feature = F.normalize(self.drug_feature, p=2, dim=1)
        self.disease_feature = F.normalize(self.disease_feature, p=2, dim=1)
        
        # Save feature shapes for model construction
        self.drug_feature_shape = self.drug_feature.shape
        self.disease_feature_shape = self.disease_feature.shape
        print("[Feature] Drug feature shape:", self.drug_feature_shape)
        print("[Feature] Disease feature shape:", self.disease_feature_shape)

    def _generate_cv_specific_graphs(self):
        """
        Generate specific graph structures for each cross-validation fold, ensuring no test edge information is included
        """
        for cv_idx in range(10):
            print(f"[Graph] Building fold {cv_idx} specific graphs...")
            
            # Get training and testing data for current fold
            train_data = self.cv_data_dict[cv_idx][0]
            test_data = self.cv_data_dict[cv_idx][1]
            
            # Create a copy of training association matrix
            train_assoc_matrix = np.zeros_like(self.association_matrix)
            
            # Fill training set edges
            pos_train_indices = train_data[train_data['values'] == 1].index
            for idx in pos_train_indices:
                drug_id = train_data.loc[idx, 'drug_id']
                disease_id = train_data.loc[idx, 'disease_id']
                train_assoc_matrix[drug_id, disease_id] = 1
            
            # Use original similarity matrices (consistent with dataloader.py)
            drug_sim_matrix = self.drug_sim_features.copy()
            disease_sim_matrix = self.disease_sim_features.copy()
            
            # Build KNN graphs based on similarity matrices
            drug_graph = self._create_similarity_graph(
                drug_sim_matrix, self.num_neighbor)
            
            disease_graph = self._create_similarity_graph(
                disease_sim_matrix, self.num_neighbor)
            
            # Build KNN graphs based on node features (using similar method as in dataloader.py)
            drug_feature_graph = self._create_feature_similarity_graph(
                'drug', self.num_neighbor)
            
            disease_feature_graph = self._create_feature_similarity_graph(
                'disease', self.num_neighbor)
            
            # Store graph structures for this fold
            self.cv_specific_graphs[cv_idx] = {
                'drug_graph': drug_graph,
                'disease_graph': disease_graph,
                'drug_feature_graph': drug_feature_graph,
                'disease_feature_graph': disease_feature_graph,
                'train_association_matrix': train_assoc_matrix
            }

    def _create_similarity_graph(self, sim_matrix, k):
        """
        Create KNN graph based on similarity matrix (similar to build_graph method in dataloader.py)
        
        Args:
          sim_matrix: Similarity matrix
          k: Number of nearest neighbors
          
        Returns:
          graph: Graph represented as torch sparse tensor
        """
        # Ensure k doesn't exceed matrix size
        k_actual = min(k, sim_matrix.shape[0] - 1)
        
        # Use KNN graph construction method similar to dataloader.py
        neighbor = np.argpartition(-sim_matrix, kth=k_actual, axis=1)[:, :k_actual]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)
        
        # Create sparse adjacency matrix
        data = np.ones(len(row_index))
        adj = sp.coo_matrix((data, (row_index, col_index)), shape=(sim_matrix.shape[0], sim_matrix.shape[0]))
        
        # Symmetrize if needed
        if self._symm:
            adj = adj + adj.T
            adj = adj.multiply(adj > 0)
            
        # Normalize and convert to torch sparse tensor
        normalized_adj = normalize(adj + sp.eye(adj.shape[0]))
        graph = sparse_mx_to_torch_sparse_tensor(normalized_adj)
        
        return graph

    def _create_feature_similarity_graph(self, node_type, k):
        """
        Create KNN graph based on node feature similarity
        
        Args:
          node_type: 'drug' or 'disease'
          k: Number of nearest neighbors
          
        Returns:
          graph: Graph represented as torch sparse tensor
        """
        # Get features
        if node_type == 'drug':
            features = self.drug_embed.copy() if hasattr(self, 'drug_embed') else self.drug_sim_features.copy()
            num_entities = self._num_drug
        else:
            features = self.disease_embed.copy() if hasattr(self, 'disease_embed') else self.disease_sim_features.copy()
            num_entities = self._num_disease
            
        # Calculate feature similarity matrix
        if len(features.shape) > 1:  # For embedding features
            # Normalize features
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized_features = features / norms
            
            # Calculate cosine similarity
            similarity_matrix = np.dot(normalized_features, normalized_features.T)
        else:  # For existing similarity matrix
            similarity_matrix = features
        
        # Use the same method as _create_similarity_graph to build KNN graph
        return self._create_similarity_graph(similarity_matrix, k)

    def _build_all_cv_data(self):
        """
        Build training and testing graphs for all cross-validation folds
        
        Returns:
          data_cv: Dictionary containing training and testing graphs for each fold
        """
        data_cv = {}
        
        for cv_idx in range(10):
            # Get training and testing data for current fold
            train_data = self.cv_data_dict[cv_idx][0]
            test_data = self.cv_data_dict[cv_idx][1]
            values = self.cv_data_dict[cv_idx][2]
            
            # Generate training and testing edges
            train_pairs, train_values = self._generate_pair_value(train_data)
            test_pairs, test_values = self._generate_pair_value(test_data)
            
            # Build encoder and decoder graphs
            train_enc_graph = self._generate_enc_graph(train_pairs, train_values, add_support=True)
            train_dec_graph = self._generate_dec_graph(train_pairs)
            
            test_enc_graph = self._generate_enc_graph(test_pairs, test_values, add_support=True)
            test_dec_graph = self._generate_dec_graph(test_pairs)
            
            # Store graph data
            data_cv[cv_idx] = {
                'train': [train_enc_graph, train_dec_graph, th.FloatTensor(train_values)],
                'test': [test_enc_graph, test_dec_graph, th.FloatTensor(test_values)]
            }
            
            print(f"[CV Data] Fold {cv_idx} train/test graphs built")
            
        return data_cv

    @staticmethod
    def _generate_pair_value(rel_info):
        """
        Generate (drug_id, disease_id) pairs and corresponding scores from DataFrame
        
        Args:
          rel_info: DataFrame containing 'drug_id', 'disease_id', 'values'
          
        Returns:
          rating_pairs, rating_values
        """
        rating_pairs = (
            np.array(rel_info["drug_id"], dtype=np.int64),
            np.array(rel_info["disease_id"], dtype=np.int64)
        )
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        """
        Construct encoder graph (DGL heterogeneous graph) containing different relations (positive/negative associations)
        
        Args:
          rating_pairs: (drug_ids, disease_ids) tuple
          rating_values: Corresponding association values
          add_support: Whether to add normalization support information
          
        Returns:
          graph: DGL heterogeneous graph
        """
        possible_rel_values = np.unique(rating_values)
        data_dict = {}
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        
        # Create mapping dictionary to map rating values to edge type names
        etype_map = {}
        rev_etype_map = {}
        
        # Ensure association type naming is consistent with original code
        print(f"[Graph] Building encoder graph with rating values: {possible_rel_values}")
        for rating in possible_rel_values:
            idx = np.where(rating_values == rating)
            rrow = rating_row[idx]
            rcol = rating_col[idx]
            
            # Use edge type naming consistent with original code
            if rating == 0:
                etype = "0"
                rev_etype = "rev-0"
            elif rating == 1:
                etype = "1"
                rev_etype = "rev-1"
            else:
                etype = str(rating).replace('.', '_')
                rev_etype = 'rev-%s' % etype
                
            # Store mapping for later use
            etype_map[rating] = etype
            rev_etype_map[rating] = rev_etype
                
            data_dict.update({
                ('drug', etype, 'disease'): (rrow, rcol),
                ('disease', rev_etype, 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # Verify edge count
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x_np = x.numpy().astype('float32')
                x_np[x_np == 0.] = np.inf
                return th.FloatTensor(1. / np.sqrt(x_np)).unsqueeze(1)
                
            drug_ci, drug_cj = [], []
            disease_ci, disease_cj = [], []
            
            for rating in possible_rel_values:
                # Use stored mapping to get correct edge type names
                etype = etype_map[rating]
                rev_etype = rev_etype_map[rating]
                
                drug_ci.append(graph[rev_etype].in_degrees())
                disease_ci.append(graph[etype].in_degrees())
                
                if self._symm:
                    drug_cj.append(graph[etype].out_degrees())
                    disease_cj.append(graph[rev_etype].out_degrees())
                else:
                    drug_cj.append(th.zeros((self._num_drug,)))
                    disease_cj.append(th.zeros((self._num_disease,)))
                    
            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self._num_drug)
                disease_cj = th.ones(self._num_disease)
                
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})
            
        return graph

    def _generate_dec_graph(self, rating_pairs):
        """
        Construct decoder graph (bipartite graph) based on drug-disease association pairs
        
        Args:
          rating_pairs: (drug_ids, disease_ids) tuple
          
        Returns:
          graph: DGL bipartite graph
        """
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self._num_drug, self._num_disease), dtype=np.float32)
            
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                              num_nodes_dict={'drug': self._num_drug, 'disease': self._num_disease})

    def augment_features(self):
        """
        Apply data augmentation to drug and disease features (noise, masking, Mixup, etc.)
        
        Returns:
          augmented_drug_feature, augmented_disease_feature
        """
        if not self.use_augmentation:
            return self.drug_feature, self.disease_feature
        
        feature_noise_scale = self.aug_params.get('feature_noise_scale', 0.05)
        feature_mask_rate = self.aug_params.get('feature_mask_rate', 0.1)
        use_mixup = self.aug_params.get('use_mixup', False)
        mixup_alpha = self.aug_params.get('mixup_alpha', 0.2)
        
        aug_drug_feature = self.drug_feature.clone()
        aug_disease_feature = self.disease_feature.clone()
        
        # Add Gaussian noise
        aug_drug_feature = GraphAugmentation.feature_noise(aug_drug_feature, feature_noise_scale)
        aug_disease_feature = GraphAugmentation.feature_noise(aug_disease_feature, feature_noise_scale)
        
        # Feature masking
        aug_drug_feature = GraphAugmentation.feature_masking(aug_drug_feature, feature_mask_rate)
        aug_disease_feature = GraphAugmentation.feature_masking(aug_disease_feature, feature_mask_rate)
        
        # Optional Mixup
        if use_mixup:
            aug_drug_feature = GraphAugmentation.mix_up_features(aug_drug_feature, mixup_alpha)
            aug_disease_feature = GraphAugmentation.mix_up_features(aug_disease_feature, mixup_alpha)
            
        return aug_drug_feature, aug_disease_feature

    def get_graph_data_for_training(self, cv_idx):
        """
        Get graph data for training at specific fold, including augmented data
        
        Args:
          cv_idx: Cross-validation fold index
          
        Returns:
          graph_data: Dictionary containing all graph data needed for training
        """
        # Get basic graph data
        cv_data = self.data_cv[cv_idx]
        cv_specific_graphs = self.cv_specific_graphs[cv_idx]
        
        # Prepare data augmentation
        if self.use_augmentation:
            aug_drug_feat, aug_disease_feat = self.augment_features()
        else:
            aug_drug_feat, aug_disease_feat = self.drug_feature, self.disease_feature
            
        # Build return graph data dictionary
        graph_data = {
            'train_enc_graph': cv_data['train'][0].to(self._device),
            'train_dec_graph': cv_data['train'][1].to(self._device),
            'train_labels': cv_data['train'][2].to(self._device),
            'test_enc_graph': cv_data['test'][0].to(self._device),
            'test_dec_graph': cv_data['test'][1].to(self._device),
            'test_labels': cv_data['test'][2].to(self._device),
            'drug_graph': cv_specific_graphs['drug_graph'].to(self._device),
            'disease_graph': cv_specific_graphs['disease_graph'].to(self._device),
            'drug_feature_graph': cv_specific_graphs['drug_feature_graph'].to(self._device),
            'disease_feature_graph': cv_specific_graphs['disease_feature_graph'].to(self._device),
            'drug_features': aug_drug_feat.to(self._device),
            'disease_features': aug_disease_feat.to(self._device),
            'drug_sim_features': th.FloatTensor(self.drug_sim_features).to(self._device),
            'disease_sim_features': th.FloatTensor(self.disease_sim_features).to(self._device)
        }
        
        return graph_data

    @property
    def num_links(self):
        """Return number of possible association values"""
        return len(np.unique(self.association_matrix))

    @property
    def num_disease(self):
        """Return number of diseases"""
        return self._num_disease

    @property
    def num_drug(self):
        """Return number of drugs"""
        return self._num_drug


if __name__ == "__main__":
    # Example: Initialize data loader and verify
    device = th.device("cpu")
    loader = DrugDataLoader(name='Gdataset', device=device, symm=True, k=5, use_augmentation=False)
    print("\n[Main] Data Loader initialization complete.")
    print(f"[Main] Number of drugs: {loader.num_drug}, Number of diseases: {loader.num_disease}")
    print("[Main] Drug feature shape:", loader.drug_feature_shape)
    print("[Main] Disease feature shape:", loader.disease_feature_shape)
    
    # Check fold 0 data
    cv_idx = 0
    graph_data = loader.get_graph_data_for_training(cv_idx)
    print(f"\n[CV Fold {cv_idx}] Train encoder graph:", graph_data['train_enc_graph'])
    print(f"[CV Fold {cv_idx}] Test encoder graph:", graph_data['test_enc_graph'])
    
    # Verify no information leakage (test)
    train_data = loader.cv_data_dict[cv_idx][0]
    test_data = loader.cv_data_dict[cv_idx][1]
    print(f"\n[Verification] Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Check no overlapping edges
    train_edges = set(zip(train_data['drug_id'], train_data['disease_id']))
    test_edges = set(zip(test_data['drug_id'], test_data['disease_id']))
    overlap = train_edges.intersection(test_edges)
    print(f"[Verification] Edge overlap between train and test: {len(overlap)} (should be 0)")