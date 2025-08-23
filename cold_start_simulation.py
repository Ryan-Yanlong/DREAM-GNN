#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cold Start Drug Simulation with Prototype Aggregation

This script implements the user's design:
1. Train model normally to get good embeddings
2. Simulate cold-start by treating masked drugs as "new drugs"
3. Use initial embeddings to compute similarity with known drugs
4. Apply top-k prototype aggregation to create "new drug" representations
5. Test how well the model can handle these simulated new drugs
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model import Net
from evaluation import evaluate
from data_loader import DrugDataLoader
from utils import MetricLogger, common_loss, setup_seed
import random


class ColdStartSimulator:
    """
    Simulates cold-start scenario by treating masked drugs as "new drugs"
    and using prototype aggregation to create their representations.
    """
    
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        
        # Store model and embeddings
        self.trained_model = None
        self.initial_drug_embeddings = None  # Initial embeddings for similarity computation
        self.trained_drug_embeddings = None  # Trained embeddings for prototype bank
        self.disease_embeddings = None
    
    def load_or_train_model(self, cv_idx):
        """Load pre-trained model or train new one."""
        # Check if pre-trained model exists
        possible_paths = []
        
        # If specific model path is provided, use it
        if hasattr(self.args, 'model_path') and self.args.model_path:
            possible_paths.append(self.args.model_path)
        
        # If model directory is provided, look for fold-specific models
        if hasattr(self.args, 'model_dir') and self.args.model_dir:
            possible_paths.extend([
                os.path.join(self.args.model_dir, f"best_model_fold{cv_idx+1}.pth"),
                os.path.join(self.args.model_dir, f"model_fold{cv_idx+1}.pth"),
                os.path.join(self.args.model_dir, f"fold{cv_idx+1}.pth")
            ])
        
        # Add default paths
        possible_paths.extend([
            f"seed_experiments/seed_42/best_model_fold{cv_idx+1}.pth",
            f"seed_experiments/seed_77/best_model_fold{cv_idx+1}.pth",
            f"seed_experiments/seed_31415/best_model_fold{cv_idx+1}.pth",
            f"best_model_fold{cv_idx+1}.pth"
        ])
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path and os.path.exists(model_path):
            print(f"=== Loading Pre-trained Model (Fold {cv_idx+1}) ===")
            return self._load_pretrained_model(cv_idx, model_path)
        else:
            print(f"=== No Pre-trained Model Found, Training New Model (Fold {cv_idx+1}) ===")
            return self._train_model_normal(cv_idx)
    
    def _load_pretrained_model(self, cv_idx, model_path):
        """Load pre-trained model and extract embeddings."""
        # Set input dimensions
        self.args.src_in_units = self.dataset.drug_feature_shape[1]
        self.args.dst_in_units = self.dataset.disease_feature_shape[1]
        self.args.fdim_drug = self.dataset.drug_feature_shape[0]
        self.args.fdim_disease = self.dataset.disease_feature_shape[0]
        self.args.rating_vals = self.dataset.cv_data_dict[cv_idx][2]
        
        # Load the saved state dict to analyze architecture
        try:
            saved_state_dict = th.load(model_path, map_location=self.device, weights_only=True)
            print(f"Analyzing saved model architecture...")
            
            # Auto-detect architecture parameters from saved model
            self._auto_detect_architecture(saved_state_dict)
            
        except Exception as e:
            print(f"Warning: Could not analyze saved model architecture: {e}")
            print("Using default architecture parameters")
        
        # Build model with detected/updated parameters
        model = Net(args=self.args).to(self.device)
        
        # Try to load pre-trained weights with strict=False to handle missing keys
        try:
            model.load_state_dict(saved_state_dict, strict=False)
            print(f"✓ Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized model")
        
        model.eval()
        
        # Extract embeddings using the loaded model
        self._extract_embeddings(model, cv_idx)
        
        # Store trained model
        self.trained_model = model
        
        # Calculate training metrics using the loaded model
        training_metrics = self._calculate_training_metrics(model, cv_idx)
        
        return model, training_metrics
    
    def _extract_embeddings(self, model, cv_idx):
        """Extract embeddings from the model."""
        # Get fold-specific data
        cv_data = self.dataset.data_cv[cv_idx]
        fold_specific_graphs = self.dataset.cv_specific_graphs[cv_idx]
        
        # Extract graph structures
        drug_graph = fold_specific_graphs['drug_graph'].to(self.device)
        dis_graph = fold_specific_graphs['disease_graph'].to(self.device)
        drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(self.device)
        disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(self.device)
        
        # Get feature data
        drug_sim_feat = th.FloatTensor(self.dataset.drug_sim_features).to(self.device)
        dis_sim_feat = th.FloatTensor(self.dataset.disease_sim_features).to(self.device)
        drug_feat = self.dataset.drug_feature.to(self.device)
        dis_feat = self.dataset.disease_feature.to(self.device)
        
        # Get training data for embedding extraction
        train_enc_graph = cv_data['train'][0].to(self.device)
        train_dec_graph = cv_data['train'][1].to(self.device)
        
        # Extract embeddings
        with th.no_grad():
            try:
                _, drug_out, drug_sim_out, dis_out, dis_sim_out = model(
                    train_enc_graph, train_dec_graph,
                    drug_graph, drug_sim_feat, drug_feat,
                    dis_graph, dis_sim_feat, dis_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
                # Store trained embeddings
                self.trained_drug_embeddings = drug_out.detach().clone()
                self.disease_embeddings = dis_out.detach().clone()
                
                # Store initial embeddings for similarity computation
                # These are the raw input features, not processed by the model
                self.initial_drug_embeddings = drug_feat.detach().clone()
                
            except Exception as e:
                print(f"Error during embedding extraction: {e}")
                print("Using fallback embedding extraction method...")
                
                # Fallback: use raw features as embeddings
                self.trained_drug_embeddings = drug_feat.detach().clone()
                self.disease_embeddings = dis_feat.detach().clone()
                self.initial_drug_embeddings = drug_feat.detach().clone()
        
        print(f"Extracted embeddings:")
        print(f"  Initial drug embeddings: {self.initial_drug_embeddings.shape}")
        print(f"  Trained drug embeddings: {self.trained_drug_embeddings.shape}")
        print(f"  Disease embeddings: {self.disease_embeddings.shape}")
    
    def _calculate_training_metrics(self, model, cv_idx):
        """Calculate training metrics using the loaded model."""
        print("Calculating training metrics...")
        
        # Get fold-specific data
        cv_data = self.dataset.data_cv[cv_idx]
        fold_specific_graphs = self.dataset.cv_specific_graphs[cv_idx]
        
        # Extract graph structures
        drug_graph = fold_specific_graphs['drug_graph'].to(self.device)
        dis_graph = fold_specific_graphs['disease_graph'].to(self.device)
        drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(self.device)
        disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(self.device)
        
        # Get feature data
        drug_sim_feat = th.FloatTensor(self.dataset.drug_sim_features).to(self.device)
        dis_sim_feat = th.FloatTensor(self.dataset.disease_sim_features).to(self.device)
        drug_feat = self.dataset.drug_feature.to(self.device)
        dis_feat = self.dataset.disease_feature.to(self.device)
        
        # Get training data for evaluation
        train_enc_graph = cv_data['train'][0].to(self.device)
        train_dec_graph = cv_data['train'][1].to(self.device)
        train_gt_ratings = cv_data['train'][2].to(self.device)
        
        model.eval()
        with th.no_grad():
            # Get predictions
            pred_ratings, _, _, _, _ = model(
                train_enc_graph, train_dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, dis_sim_feat, dis_feat,
                drug_feature_graph, disease_feature_graph
            )
            
            # Calculate metrics
            pred_scores = th.sigmoid(pred_ratings.squeeze(-1))
            
            # Convert to numpy for metric calculation
            pred_scores_np = pred_scores.cpu().numpy()
            gt_ratings_np = train_gt_ratings.cpu().numpy()
            
            # Calculate AUROC and AUPR using sklearn
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                auroc = roc_auc_score(gt_ratings_np, pred_scores_np)
                aupr = average_precision_score(gt_ratings_np, pred_scores_np)
            except ImportError:
                print("Warning: sklearn not available, using default metrics")
                auroc = 0.5
                aupr = 0.5
        
        print(f"Training metrics - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
        return {'auroc': auroc, 'aupr': aupr}
    
    def _auto_detect_architecture(self, saved_state_dict):
        """Auto-detect model architecture parameters from saved state dict."""
        print("  Auto-detecting architecture parameters...")
        
        # Check FGCN dimensions
        if 'FGCN.FGCN_drug.gc1.weight' in saved_state_dict:
            drug_gc1_weight = saved_state_dict['FGCN.FGCN_drug.gc1.weight']
            detected_nhid1 = drug_gc1_weight.shape[1]
            if detected_nhid1 != self.args.nhid1:
                print(f"    Updating nhid1: {self.args.nhid1} -> {detected_nhid1}")
                self.args.nhid1 = detected_nhid1
        
        if 'FGCN.FGCN_drug.gc2.weight' in saved_state_dict:
            drug_gc2_weight = saved_state_dict['FGCN.FGCN_drug.gc2.weight']
            detected_nhid2 = drug_gc2_weight.shape[1]
            if detected_nhid2 != self.args.nhid2:
                print(f"    Updating nhid2: {self.args.nhid2} -> {detected_nhid2}")
                self.args.nhid2 = detected_nhid2
        
        # Check TGCN dimensions
        if 'TGCN.0.basis' in saved_state_dict:
            tgcn_basis = saved_state_dict['TGCN.0.basis']
            detected_basis_units = tgcn_basis.shape[0]
            if hasattr(self.args, 'basis_units') and detected_basis_units != self.args.basis_units:
                print(f"    Updating basis_units: {self.args.basis_units} -> {detected_basis_units}")
                self.args.basis_units = detected_basis_units
        
        print(f"  Architecture parameters updated:")
        print(f"    - nhid1: {self.args.nhid1}")
        print(f"    - nhid2: {self.args.nhid2}")
        if hasattr(self.args, 'basis_units'):
            print(f"    - basis_units: {self.args.basis_units}")
    
    def _train_model_normal(self, cv_idx):
        """Train the model normally."""
        print(f"=== Training Model Normally (Fold {cv_idx+1}) ===")
        
        # Set input dimensions
        self.args.src_in_units = self.dataset.drug_feature_shape[1]
        self.args.dst_in_units = self.dataset.disease_feature_shape[1]
        self.args.fdim_drug = self.dataset.drug_feature_shape[0]
        self.args.fdim_disease = self.dataset.disease_feature_shape[0]
        self.args.rating_vals = self.dataset.cv_data_dict[cv_idx][2]
        
        # Get fold-specific data
        cv_data = self.dataset.data_cv[cv_idx]
        fold_specific_graphs = self.dataset.cv_specific_graphs[cv_idx]
        
        # Extract graph structures
        drug_graph = fold_specific_graphs['drug_graph'].to(self.device)
        dis_graph = fold_specific_graphs['disease_graph'].to(self.device)
        drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(self.device)
        disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(self.device)
        
        # Get feature data
        drug_sim_feat = th.FloatTensor(self.dataset.drug_sim_features).to(self.device)
        dis_sim_feat = th.FloatTensor(self.dataset.disease_sim_features).to(self.device)
        drug_feat = self.dataset.drug_feature.to(self.device)
        dis_feat = self.dataset.disease_feature.to(self.device)
        
        # Get training data
        train_gt_ratings = cv_data['train'][2].to(self.device)
        train_enc_graph = cv_data['train'][0].to(self.device)
        train_dec_graph = cv_data['train'][1].to(self.device)
        
        # Build model
        model = Net(args=self.args).to(self.device)
        
        # Use standard BCE loss
        criterion = nn.BCEWithLogitsLoss()
        optimizer = th.optim.Adam(model.parameters(), lr=self.args.train_lr)
        
        print("Training model normally...")
        best_aupr = -1.0
        best_auroc = 0.0
        
        # Training loop
        for epoch in range(1000):  # Reduced epochs for demonstration
            model.train()
            
            # Forward pass
            pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out = model(
                train_enc_graph, train_dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, dis_sim_feat, dis_feat,
                drug_feature_graph, disease_feature_graph
            )
            
            # Loss
            loss = criterion(pred_ratings.squeeze(-1), train_gt_ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate periodically
            if epoch % 100 == 0:
                model.eval()
                with th.no_grad():
                    # Get predictions for evaluation
                    pred_ratings_eval, _, _, _, _ = model(
                        train_enc_graph, train_dec_graph,
                        drug_graph, drug_sim_feat, drug_feat,
                        dis_graph, dis_sim_feat, dis_feat,
                        drug_feature_graph, disease_feature_graph
                    )
                    
                    # Calculate metrics
                    pred_scores = th.sigmoid(pred_ratings_eval.squeeze(-1))
                    
                    # Convert to numpy for metric calculation
                    pred_scores_np = pred_scores.cpu().numpy()
                    gt_ratings_np = train_gt_ratings.cpu().numpy()
                    
                    # Calculate AUROC and AUPR using sklearn
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        auroc = roc_auc_score(gt_ratings_np, pred_scores_np)
                        aupr = average_precision_score(gt_ratings_np, pred_scores_np)
                    except ImportError:
                        print("Warning: sklearn not available, using default metrics")
                        auroc = 0.5
                        aupr = 0.5
                    
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
                    
                    # Store best embeddings based on AUPR
                    if aupr > best_aupr:
                        best_aupr = aupr
                        best_auroc = auroc
                        
                        # Store the best embeddings
                        self.trained_drug_embeddings = drug_out.detach().clone()
                        self.disease_embeddings = dis_out.detach().clone()
                        
                        # Store initial embeddings for similarity computation
                        self.initial_drug_embeddings = drug_feat.detach().clone()
        
        print(f"Training completed. Best accuracy: {best_aupr:.4f}")
        
        # Store trained model
        self.trained_model = model
        
        return model, {'auroc': best_auroc, 'aupr': best_aupr}
    
    def select_cold_start_drugs(self, cold_start_ratio=0.1):
        """
        Select drugs to simulate as cold-start drugs.
        These are drugs that will be treated as "new drugs".
        
        Args:
            cold_start_ratio: Ratio of drugs to treat as cold-start
            
        Returns:
            cold_start_drugs: List of drug indices to simulate as new drugs
            known_drugs: List of drug indices that remain as known drugs
        """
        print(f"=== Selecting Cold-Start Drugs (Ratio: {cold_start_ratio}) ===")
        
        if self.initial_drug_embeddings is None:
            raise ValueError("Must load or train model first to get embeddings")
        
        num_drugs = self.initial_drug_embeddings.shape[0]
        num_cold_start = max(1, int(num_drugs * cold_start_ratio))
        
        # Randomly select drugs to simulate as cold-start
        all_drugs = list(range(num_drugs))
        cold_start_drugs = random.sample(all_drugs, num_cold_start)
        known_drugs = [d for d in all_drugs if d not in cold_start_drugs]
        
        print(f"Selected {num_cold_start} drugs as cold-start drugs: {cold_start_drugs[:10]}...")
        print(f"Remaining {len(known_drugs)} drugs as known drugs")
        
        return cold_start_drugs, known_drugs
    
    def compute_similarity_and_aggregate(self, cold_start_drugs, known_drugs):
        """
        Compute similarity between cold-start drugs and known drugs using INITIAL embeddings,
        then aggregate to create representations for cold-start drugs.
        
        Args:
            cold_start_drugs: List of drug indices to simulate as new drugs
            known_drugs: List of drug indices that remain as known drugs
            
        Returns:
            aggregated_embeddings: Prototype-aggregated embeddings for cold-start drugs
            selected_known_drugs: List of selected known drug indices for each cold-start drug
            weights: Softmax weights for each cold-start drug
        """
        print(f"=== Computing Similarity and Aggregating ===")
        
        if self.initial_drug_embeddings is None or self.trained_drug_embeddings is None:
            raise ValueError("Must load or train model first to get embeddings")
        
        # Get initial embeddings for similarity computation (lrssl原始embedding)
        cold_start_initial = self.initial_drug_embeddings[cold_start_drugs]  # [num_cold_start, 768]
        known_initial = self.initial_drug_embeddings[known_drugs]           # [num_known, 768]
        
        # Get trained embeddings for aggregation (网络输出的embedding)
        known_trained = self.trained_drug_embeddings[known_drugs]  # [num_known, 128]
        
        print(f"Cold-start initial embeddings (lrssl): {cold_start_initial.shape}")
        print(f"Known initial embeddings (lrssl): {known_initial.shape}")
        print(f"Known trained embeddings (network output): {known_trained.shape}")
        
        # Compute similarity between cold-start and known drugs using INITIAL embeddings
        # Use cosine similarity
        cold_start_norm = F.normalize(cold_start_initial, p=2, dim=1)
        known_norm = F.normalize(known_initial, p=2, dim=1)
        
        # Similarity matrix: [num_cold_start, num_known]
        similarities = th.mm(cold_start_norm, known_norm.t())
        
        print(f"Similarity matrix shape: {similarities.shape}")
        print(f"Similarity range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
        
        # Get top-k most similar known drugs for each cold-start drug
        k = min(self.args.prototype_k, len(known_drugs))
        top_k_similarities, top_k_indices = th.topk(similarities, k=k, dim=1)
        
        print(f"Top-{k} similarities range: [{top_k_similarities.min().item():.4f}, {top_k_similarities.max().item():.4f}]")
        
        # Apply softmax to get weights
        scaled_similarities = top_k_similarities / self.args.prototype_temperature
        weights = F.softmax(scaled_similarities, dim=1)
        
        print(f"Weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
        print(f"Average weight: {weights.mean().item():.4f}")
        
        # Aggregate trained embeddings using weights
        aggregated_embeddings = []
        selected_known_drugs = []
        
        for i in range(len(cold_start_drugs)):
            # Get top-k prototype embeddings for this cold-start drug
            drug_top_k_indices = top_k_indices[i]  # [k]
            drug_weights = weights[i]              # [k]
            
            # Get corresponding trained embeddings from known drugs
            drug_prototypes = known_trained[drug_top_k_indices]  # [k, 128]
            
            # Weighted aggregation: sum(weight * trained_embedding)
            weighted_sum = th.sum(drug_weights.unsqueeze(-1) * drug_prototypes, dim=0)  # [128]
            aggregated_embeddings.append(weighted_sum)
            
            # Record selected known drug indices
            selected_known_drugs.append(drug_top_k_indices.cpu().numpy().tolist())
        
        aggregated_embeddings = th.stack(aggregated_embeddings, dim=0)  # [num_cold_start, 128]
        
        print(f"Aggregated embeddings shape: {aggregated_embeddings.shape}")
        
        return aggregated_embeddings, similarities, weights, selected_known_drugs
    
    def test_cold_start_performance(self, cold_start_drugs, aggregated_embeddings, cv_idx):
        """
        Test how well the model performs on cold-start drugs using aggregated embeddings.
        
        Args:
            cold_start_drugs: List of cold-start drug indices
            aggregated_embeddings: Prototype-aggregated embeddings for cold-start drugs
            cv_idx: Cross-validation fold index
            
        Returns:
            cold_start_auroc: AUROC on cold-start drugs
            cold_start_aupr: AUPR on cold-start drugs
            detailed_results: Detailed results
        """
        print(f"=== Testing Cold-Start Performance ===")
        
        if self.trained_model is None:
            raise ValueError("Must load or train model first")
        
        # Get test data for this fold
        cv_data = self.dataset.data_cv[cv_idx]
        test_data = cv_data['test']
        
        # Create a simple MLP for cold-start testing
        # Calculate input dimension based on actual embedding dimensions
        drug_emb_dim = aggregated_embeddings.shape[1]
        disease_emb_dim = self.disease_embeddings.shape[1] # Use actual disease embedding dimension
        input_dim = drug_emb_dim + disease_emb_dim
        
        cold_start_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Train cold-start MLP using aggregated embeddings
        print("Training cold-start MLP with aggregated embeddings...")
        
        # Prepare training data using real associations from the dataset
        train_pairs = []
        train_labels = []
        
        # Get all disease embeddings
        disease_embs = self.disease_embeddings
        
        # For each cold-start drug, find its real associations in the training data
        print(f"Processing {len(cold_start_drugs)} cold-start drugs for training data...")
        for i, drug_idx in enumerate(cold_start_drugs):
            drug_emb = aggregated_embeddings[i:i+1]
            print(f"Processing cold-start drug {drug_idx} ({i+1}/{len(cold_start_drugs)})")
            
            # Find real associations for this drug in the training data
            train_enc_graph = cv_data['train'][0]  # DGL graph object
            train_gt_ratings = cv_data['train'][2]  # Tensor of ratings
            
            # Find edges where this drug appears
            if hasattr(train_enc_graph, 'edges'):
                # Get edge indices where this drug is the source
                # Handle heterogeneous DGL graphs with proper edge type specification
                try:
                    # For heterogeneous graphs, we need to specify the edge type
                    # Try to get the canonical edge types first
                    if hasattr(train_enc_graph, 'canonical_etypes'):
                        canonical_etypes = train_enc_graph.canonical_etypes
                        print(f"  Graph has canonical_etypes: {canonical_etypes}")
                        if len(canonical_etypes) > 0:
                            # Use the first edge type (usually the main one)
                            edge_type = canonical_etypes[0]
                            src_nodes, dst_nodes = train_enc_graph.edges(etype=edge_type)
                            print(f"  Using edge type {edge_type}, found {len(src_nodes)} edges")
                        else:
                            # Try without specifying edge type for homogeneous graphs
                            src_nodes, dst_nodes = train_enc_graph.edges()
                            print(f"  No edge types, found {len(src_nodes)} edges")
                    else:
                        # Try without specifying edge type for homogeneous graphs
                        src_nodes, dst_nodes = train_enc_graph.edges()
                        print(f"  No canonical_etypes, found {len(src_nodes)} edges")
                except Exception as e:
                    print(f"Warning: Could not access edges with edge type: {e}")
                    # Fallback: try to access edges with 'all' parameter
                    try:
                        src_nodes, dst_nodes = train_enc_graph.edges(form='uv')
                        print(f"  Fallback successful, found {len(src_nodes)} edges")
                    except Exception as e2:
                        print(f"Warning: Fallback edge access also failed: {e2}")
                        continue
                
                # Find edges where this drug appears as source
                drug_edge_mask = (src_nodes == drug_idx)
                drug_edge_indices = drug_edge_mask.nonzero(as_tuple=True)[0]
                
                if len(drug_edge_indices) > 0:
                    print(f"  Found {len(drug_edge_indices)} edges for drug {drug_idx}")
                    for edge_idx in drug_edge_indices:
                        disease_idx = dst_nodes[edge_idx].item()
                        label = train_gt_ratings[edge_idx].item()
                        
                        # Get disease embedding
                        disease_emb = disease_embs[disease_idx:disease_idx+1]
                        combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                        
                        # Only add if the combined embedding is valid
                        if combined_emb.numel() > 0:
                            train_pairs.append(combined_emb)
                            train_labels.append(label)
                else:
                    print(f"  No edges found for drug {drug_idx}")
            else:
                print(f"Warning: Unexpected training graph structure for drug {drug_idx}")
                
            # If no real associations found, create some synthetic ones for training stability
            current_pairs_for_drug = len([p for p in train_pairs if p.shape[0] > 0]) if train_pairs else 0
            if current_pairs_for_drug == len(train_pairs) - len(cold_start_drugs) + i:  # No new pairs added
                print(f"  Adding synthetic data for drug {drug_idx}")
                # Create a few synthetic positive and negative samples
                num_synthetic = 5
                positive_diseases = random.sample(range(disease_embs.shape[0]), min(num_synthetic, disease_embs.shape[0]))
                
                for disease_idx in positive_diseases:
                    disease_emb = disease_embs[disease_idx:disease_idx+1]
                    combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                    train_pairs.append(combined_emb)
                    train_labels.append(1.0)
                
                negative_diseases = random.sample(range(disease_embs.shape[0]), min(num_synthetic, disease_embs.shape[0]))
                for disease_idx in negative_diseases:
                    disease_emb = disease_embs[disease_idx:disease_idx+1]
                    combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                    train_pairs.append(combined_emb)
                    train_labels.append(0.0)

        
        # Stack all training data
        print(f"Total training pairs collected: {len(train_pairs)}")
        print(f"Total training labels collected: {len(train_labels)}")
        
        if train_pairs:
            # Filter out empty tensors and corresponding labels before concatenation
            valid_pairs = []
            valid_labels = []
            for i, pair in enumerate(train_pairs):
                if pair.numel() > 0:
                    valid_pairs.append(pair)
                    valid_labels.append(train_labels[i])
            
            print(f"Valid pairs after filtering: {len(valid_pairs)}")
            print(f"Valid labels after filtering: {len(valid_labels)}")
            
            if valid_pairs:
                train_pairs = th.cat(valid_pairs, dim=0)
                train_labels = th.FloatTensor(valid_labels).to(self.device)
                print(f"Final training data shape: {train_pairs.shape}")
                print(f"Final training labels shape: {train_labels.shape}")
            else:
                print("Warning: No valid training pairs found, using synthetic data")
                # Create synthetic training data if no valid pairs
                num_synthetic = 10
                positive_diseases = random.sample(range(disease_embs.shape[0]), min(num_synthetic, disease_embs.shape[0]))
                
                for disease_idx in positive_diseases:
                    disease_emb = disease_embs[disease_idx:disease_idx+1]
                    # Use the first cold-start drug for synthetic data
                    drug_emb = aggregated_embeddings[0:1]
                    combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                    train_pairs = [combined_emb]
                    train_labels = [1.0]
                
                train_pairs = th.cat(train_pairs, dim=0)
                train_labels = th.FloatTensor(train_labels).to(self.device)
            
            # Train cold-start MLP
            criterion = nn.BCEWithLogitsLoss()
            optimizer = th.optim.Adam(cold_start_mlp.parameters(), lr=0.001)
            
            cold_start_mlp.train()
            for epoch in range(100):
                optimizer.zero_grad()
                
                pred_scores = cold_start_mlp(train_pairs).squeeze(-1)
                loss = criterion(pred_scores, train_labels)
                
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    print(f"Cold-start MLP Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Test cold-start performance using REAL test data
        print("Testing cold-start performance...")
        cold_start_mlp.eval()
        
        # Get test data - these are DGL graph objects, not simple tensors
        test_enc_graph = test_data[0]  # DGL graph object
        test_gt_ratings = test_data[2]  # Tensor of ratings
        
        # Find test edges involving cold-start drugs
        cold_start_test_edges = []
        cold_start_test_labels = []
        cold_start_test_predictions = []
        
        with th.no_grad():
            for i, drug_idx in enumerate(cold_start_drugs):
                drug_emb = aggregated_embeddings[i:i+1]
                
                # Find test edges where this cold-start drug appears
                # For DGL graphs, we need to access the source nodes
                if hasattr(test_enc_graph, 'edges'):
                    # Get edge indices where this drug is the source
                    # Handle heterogeneous DGL graphs with proper edge type specification
                    try:
                        # For heterogeneous graphs, we need to specify the edge type
                        # Try to get the canonical edge types first
                        if hasattr(test_enc_graph, 'canonical_etypes'):
                            canonical_etypes = test_enc_graph.canonical_etypes
                            if len(canonical_etypes) > 0:
                                # Use the first edge type (usually the main one)
                                edge_type = canonical_etypes[0]
                                src_nodes, dst_nodes = test_enc_graph.edges(etype=edge_type)
                            else:
                                # Try without specifying edge type for homogeneous graphs
                                src_nodes, dst_nodes = test_enc_graph.edges()
                        else:
                            # Try without specifying edge type for homogeneous graphs
                            src_nodes, dst_nodes = test_enc_graph.edges()
                    except Exception as e:
                        print(f"Warning: Could not access test edges with edge type: {e}")
                        # Fallback: try to access edges with 'all' parameter
                        try:
                            src_nodes, dst_nodes = test_enc_graph.edges(form='uv')
                        except Exception as e2:
                            print(f"Warning: Fallback test edge access also failed: {e2}")
                            continue
                    
                    # Find edges where this cold-start drug appears as source
                    drug_test_edge_mask = (src_nodes == drug_idx)
                    drug_test_edge_indices = drug_test_edge_mask.nonzero(as_tuple=True)[0]
                    
                    for edge_idx in drug_test_edge_indices:
                        disease_idx = dst_nodes[edge_idx].item()
                        true_label = test_gt_ratings[edge_idx].item()
                        
                        # Get disease embedding
                        disease_emb = disease_embs[disease_idx:disease_idx+1]
                        combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                        
                        # Predict
                        pred_score = cold_start_mlp(combined_emb)
                        pred_prob = th.sigmoid(pred_score).item()
                        
                        cold_start_test_edges.append((drug_idx, disease_idx))
                        cold_start_test_labels.append(true_label)
                        cold_start_test_predictions.append(pred_prob)
                else:
                    # Fallback: if graph structure is different, create synthetic test data
                    print(f"Warning: Unexpected graph structure for drug {drug_idx}, using synthetic test data")
                    num_synthetic_test = 5
                    test_diseases = random.sample(range(disease_embs.shape[0]), min(num_synthetic_test, disease_embs.shape[0]))
                    
                    for disease_idx in test_diseases:
                        disease_emb = disease_embs[disease_idx:disease_idx+1]
                        combined_emb = th.cat([drug_emb, disease_emb], dim=1)
                        
                        # Predict
                        pred_score = cold_start_mlp(combined_emb)
                        pred_prob = th.sigmoid(pred_score).item()
                        
                        cold_start_test_edges.append((drug_idx, disease_idx))
                        cold_start_test_labels.append(0.5)  # Synthetic label
                        cold_start_test_predictions.append(pred_prob)
        
        # Calculate success rate (how many predictions are correct)
        if cold_start_test_labels:
            # Convert to numpy for metric calculation
            test_labels_np = np.array(cold_start_test_labels)
            test_predictions_np = np.array(cold_start_test_predictions)
            
            # Calculate success rate (predictions > 0.5 for positive samples, < 0.5 for negative)
            predictions_binary = (test_predictions_np > 0.5).astype(float)
            success_count = np.sum(predictions_binary == test_labels_np)
            total_tested = len(test_labels_np)
            
            # Calculate AUROC and AUPR
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                cold_start_auroc = roc_auc_score(test_labels_np, test_predictions_np)
                cold_start_aupr = average_precision_score(test_labels_np, test_predictions_np)
            except ImportError:
                print("Warning: sklearn not available, using default metrics")
                cold_start_auroc = 0.5
                cold_start_aupr = 0.5
            
            cold_start_success_rate = success_count / total_tested
            
            print(f"Cold-Start Test Results:")
            print(f"  Total test edges: {total_tested}")
            print(f"  Success count: {success_count}")
            print(f"  Success rate: {cold_start_success_rate:.4f}")
            print(f"  AUROC: {cold_start_auroc:.4f}")
            print(f"  AUPR: {cold_start_aupr:.4f}")
            
        else:
            print("No test edges found for cold-start drugs")
            cold_start_success_rate = 0.0
            cold_start_auroc = 0.0
            cold_start_aupr = 0.0
            total_tested = 0
        
        # Create detailed results
        detailed_results = []
        for i, (drug_idx, disease_idx) in enumerate(cold_start_test_edges):
            detailed_results.append({
                'drug_id': drug_idx,
                'disease_id': disease_idx,
                'true_label': cold_start_test_labels[i],
                'prediction': cold_start_test_predictions[i],
                'success': abs(cold_start_test_labels[i] - cold_start_test_predictions[i]) < 0.5
            })
        
        return cold_start_success_rate, detailed_results
    
    def run_cold_start_experiment(self, cv_idx, cold_start_ratio=0.1):
        """
        Run the complete cold-start simulation experiment.
        
        Args:
            cv_idx: Cross-validation fold index
            cold_start_ratio: Ratio of drugs to simulate as cold-start
            
        Returns:
            results: Complete experiment results
        """
        print(f"\n{'='*60}")
        print(f"COLD-START SIMULATION EXPERIMENT - Fold {cv_idx+1}")
        print(f"{'='*60}")
        
        # Step 1: Load or train model
        trained_model, training_metrics = self.load_or_train_model(cv_idx)
        
        # Step 2: Select cold-start drugs
        cold_start_drugs, known_drugs = self.select_cold_start_drugs(cold_start_ratio)
        
        # Step 3: Compute similarity and aggregate
        aggregated_embeddings, similarities, weights, selected_known_drugs = self.compute_similarity_and_aggregate(
            cold_start_drugs, known_drugs
        )
        
        # Step 4: Test cold-start performance
        cold_start_success_rate, detailed_results = self.test_cold_start_performance(
            cold_start_drugs, aggregated_embeddings, cv_idx
        )
        
        # Compile results
        results = {
            'fold': cv_idx + 1,
            'training_metrics': training_metrics,
            'cold_start_ratio': cold_start_ratio,
            'num_cold_start_drugs': len(cold_start_drugs),
            'num_known_drugs': len(known_drugs),
            'cold_start_success_rate': cold_start_success_rate,
            'detailed_results': detailed_results,
            'prototype_k': self.args.prototype_k,
            'prototype_temperature': self.args.prototype_temperature,
            'selected_known_drugs': selected_known_drugs,  # 记录每个cold start药物选择的已知药物
            'similarity_stats': {
                'min': similarities.min().item(),
                'max': similarities.max().item(),
                'mean': similarities.mean().item(),
                'std': similarities.std().item()
            },
            'weight_stats': {
                'min': weights.min().item(),
                'max': weights.max().item(),
                'mean': weights.mean().item(),
                'std': weights.std().item()
            }
        }
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RESULTS - Fold {cv_idx+1}")
        print(f"{'='*60}")
        print(f"Training AUROC: {training_metrics['auroc']:.4f}")
        print(f"Training AUPR: {training_metrics['aupr']:.4f}")
        print(f"Cold-Start Success Rate: {cold_start_success_rate:.4f}")
        print(f"Cold-Start Drugs: {len(cold_start_drugs)}")
        print(f"Known Drugs: {len(known_drugs)}")
        print(f"Prototype k: {self.args.prototype_k}")
        print(f"Prototype Temperature: {self.args.prototype_temperature}")
        print(f"Similarity Range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
        print(f"Weight Range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
        
        # Display selected known drugs for each cold-start drug
        print(f"\nSelected Known Drugs for Cold-Start Drugs:")
        for i, drug_idx in enumerate(cold_start_drugs):
            selected_drugs = selected_known_drugs[i]
            print(f"  Cold-Start Drug {drug_idx}: Selected Known Drugs {selected_drugs[:5]}... (weights: {weights[i][:3].cpu().numpy()})")
        
        return results


def main():
    """Main function to run the cold-start simulation experiment."""
    parser = argparse.ArgumentParser(description='Cold-Start Drug Simulation Experiment')
    parser.add_argument('--data_name', type=str, default='lrssl', help='Dataset name')
    parser.add_argument('--device', type=int, default=-1, help='Device to use (-1 for CPU)')
    parser.add_argument('--prototype_k', type=int, default=5, help='Number of top prototypes')
    parser.add_argument('--prototype_temperature', type=float, default=1.0, help='Temperature for softmax')
    parser.add_argument('--cold_start_ratio', type=float, default=0.1, help='Ratio of drugs to simulate as cold-start')
    parser.add_argument('--train_lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--gcn_out_units', type=int, default=128, help='GCN output units')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--gcn_agg_units', type=int, default=1024, help='GCN aggregation units')
    parser.add_argument('--nhid1', type=int, default=768, help='Hidden layer 1 units')
    parser.add_argument('--nhid2', type=int, default=128, help='Hidden layer 2 units')
    parser.add_argument('--fdim_drug', type=int, default=128, help='Drug feature dimension')
    parser.add_argument('--fdim_disease', type=int, default=128, help='Disease feature dimension')
    parser.add_argument('--gcn_agg_accum', type=str, default='sum', help='GCN aggregation accumulation')
    parser.add_argument('--share_param', default=True, action='store_true', help='Share parameters')
    parser.add_argument('--model_activation', type=str, default='leaky', help='Model activation')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Attention dropout')
    parser.add_argument('--rating_vals', type=list, default=[0, 1], help='Rating values')
    parser.add_argument('--basis_units', type=int, default=2, help='Basis units for TGCN')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model file')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory containing model files')
    
    args = parser.parse_args()
    
    # Set device
    if args.device >= 0:
        args.device = f"cuda:{args.device}" if th.cuda.is_available() else "cpu"
    else:
        args.device = "cpu"
    print(f"Using device: {args.device}")
    
    # Set random seed
    setup_seed(42)
    
    # Initialize data loader
    dataset = DrugDataLoader(args.data_name, args.device, symm=True, k=5)
    print("Dataset loaded successfully.")
    
    # Create experiment runner
    experiment_runner = ColdStartSimulator(args, dataset)
    
    # Run experiment for each fold
    all_results = []
    for cv_idx in range(5):  # Test with 5 folds for demonstration
        try:
            results = experiment_runner.run_cold_start_experiment(
                cv_idx, 
                cold_start_ratio=args.cold_start_ratio
            )
            all_results.append(results)
            print(f"Fold {cv_idx+1} completed successfully.")
        except Exception as e:
            print(f"Error in fold {cv_idx+1}: {str(e)}")
            continue
    
    # Compile final results
    if all_results:
        print(f"\n{'='*60}")
        print("FINAL EXPERIMENT RESULTS")
        print(f"{'='*60}")
        
        # Calculate average cold-start success rate
        success_rates = [r['cold_start_success_rate'] for r in all_results]
        avg_success_rate = np.mean(success_rates)
        std_success_rate = np.std(success_rates)
        
        print(f"Average Cold-Start Success Rate: {avg_success_rate:.4f} ± {std_success_rate:.4f}")
        print(f"Individual Fold Results:")
        
        for result in all_results:
            print(f"  Fold {result['fold']}: {result['cold_start_success_rate']:.4f}")
        
        # Save results
        output_file = f"cold_start_simulation_results_{args.data_name}.csv"
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'fold': result['fold'],
                'cold_start_success_rate': result['cold_start_success_rate'],
                'num_cold_start_drugs': result['num_cold_start_drugs'],
                'num_known_drugs': result['num_known_drugs'],
                'prototype_k': result['prototype_k'],
                'prototype_temperature': result['prototype_temperature'],
                'selected_known_drugs': str(result['selected_known_drugs']),  # 保存选中的已知药物信息
                'similarity_min': result['similarity_stats']['min'],
                'similarity_max': result['similarity_stats']['max'],
                'similarity_mean': result['similarity_stats']['mean'],
                'weight_min': result['weight_stats']['min'],
                'weight_max': result['weight_stats']['max'],
                'weight_mean': result['weight_stats']['mean']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Show prototype aggregation parameters
        print(f"\nPrototype Aggregation Parameters:")
        print(f"  k: {args.prototype_k}")
        print(f"  Temperature: {args.prototype_temperature}")
        print(f"  Cold-Start Ratio: {args.cold_start_ratio}")
        
    else:
        print("No results obtained from any fold.")


if __name__ == '__main__':
    main()
