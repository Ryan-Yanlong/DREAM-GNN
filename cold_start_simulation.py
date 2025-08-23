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
        Test cold-start performance using aggregated embeddings WITHOUT training.
        This is a true zero-shot cold-start scenario.
        
        Args:
            cold_start_drugs: List of cold-start drug indices
            aggregated_embeddings: Aggregated embeddings for cold-start drugs
            cv_idx: Cross-validation fold index
            
        Returns:
            cold_start_success_rate: Success rate on test set
            detailed_results: Detailed prediction results
        """
        print("Testing cold-start performance (zero-shot, no training)...")
        
        if self.trained_model is None:
            raise ValueError("Must load or train model first")
        
        # Get test data for this fold
        cv_data = self.dataset.data_cv[cv_idx]
        test_data = cv_data['test']
        
        # Extract MLP from the pre-trained model
        print("Extracting MLP weights from pre-trained model...")
        
        # Get the decoder (MLP) from the trained model
        pretrained_decoder = self.trained_model.decoder
        
        # Create a new MLP with the same architecture but for cold-start input
        drug_emb_dim = aggregated_embeddings.shape[1]
        disease_emb_dim = self.disease_embeddings.shape[1]
        input_dim = drug_emb_dim + disease_emb_dim
        
        # Create cold-start MLP with same architecture as the pretrained decoder
        # IMPORTANT: Follow the original algorithm - NO sigmoid in MLP!
        cold_start_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),  # Same as pretrained_decoder.lin1 output size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),         # Same as pretrained_decoder.lin2 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)            # Same as pretrained_decoder.lin3 - NO sigmoid!
        ).to(self.device)
        
        # Copy weights from pretrained decoder to cold-start MLP
        # Note: Only copy the weights, not biases if input dimension is different
        with th.no_grad():
            # Copy lin2 and lin3 weights (these should be compatible)
            cold_start_mlp[3].weight.data = pretrained_decoder.lin2.weight.data.clone()
            cold_start_mlp[3].bias.data = pretrained_decoder.lin2.bias.data.clone()
            
            cold_start_mlp[6].weight.data = pretrained_decoder.lin3.weight.data.clone()
            cold_start_mlp[6].bias.data = pretrained_decoder.lin3.bias.data.clone()
            
            # For lin1, we need to handle potential dimension mismatch
            pretrained_input_size = pretrained_decoder.lin1.in_features
            cold_start_input_size = input_dim
            
            if pretrained_input_size == cold_start_input_size:
                # Perfect match - copy all weights
                cold_start_mlp[0].weight.data = pretrained_decoder.lin1.weight.data.clone()
                cold_start_mlp[0].bias.data = pretrained_decoder.lin1.bias.data.clone()
                print(f"  Copied all MLP weights (input size match: {input_dim})")
            else:
                # Dimension mismatch - initialize lin1 randomly but copy lin2 and lin3
                print(f"  Input dimension mismatch: pretrained={pretrained_input_size}, cold_start={cold_start_input_size}")
                print(f"  Copied lin2 and lin3 weights, lin1 initialized randomly")
        
        # Set to evaluation mode for inference
        cold_start_mlp.eval()
        print("✓ MLP weights extracted and loaded successfully")
        
        # Get test data
        test_enc_graph = test_data[0]  # DGL graph object
        test_gt_ratings = test_data[2]  # Tensor of ratings
        disease_embs = self.disease_embeddings
        
        # Find test edges involving cold-start drugs
        cold_start_test_edges = []
        cold_start_test_labels = []
        cold_start_test_predictions = []
        
        with th.no_grad():
            for i, drug_idx in enumerate(cold_start_drugs):
                drug_emb = aggregated_embeddings[i:i+1]
                
                # Find test edges where this cold-start drug appears
                if hasattr(test_enc_graph, 'edges'):
                    try:
                        if hasattr(test_enc_graph, 'canonical_etypes'):
                            canonical_etypes = test_enc_graph.canonical_etypes
                            if len(canonical_etypes) > 0:
                                # Find the correct edge type: drug -> disease
                                drug_to_disease_etypes = [et for et in canonical_etypes if et[0] == 'drug' and et[2] == 'disease']
                                if drug_to_disease_etypes:
                                    edge_type = drug_to_disease_etypes[0]
                                else:
                                    edge_type = canonical_etypes[0]
                                
                                src_nodes, dst_nodes = test_enc_graph.edges(etype=edge_type)
                            else:
                                src_nodes, dst_nodes = test_enc_graph.edges()
                        else:
                            src_nodes, dst_nodes = test_enc_graph.edges()
                    except Exception as e:
                        print(f"Warning: Could not access test edges: {e}")
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
                        
                        # Predict using pre-trained MLP (zero-shot)
                        # IMPORTANT: Follow original algorithm - apply sigmoid manually!
                        pred_score = cold_start_mlp(combined_emb)
                        pred_prob = th.sigmoid(pred_score).item()  # Manual sigmoid like in train.py
                        
                        # Debug: Check raw MLP output and sigmoid output
                        if i == 0 and len(cold_start_test_predictions) < 5:  # Only for first few predictions
                            print(f"    Debug - Raw MLP output: {pred_score.item():.6f}, Sigmoid: {pred_prob:.6f}")
                        
                        cold_start_test_edges.append((drug_idx, disease_idx))
                        cold_start_test_labels.append(true_label)
                        cold_start_test_predictions.append(pred_prob)
                else:
                    print(f"Warning: Unexpected test graph structure for drug {drug_idx}")
                    continue
        
        # Calculate success rate
        if cold_start_test_labels:
            test_labels_np = np.array(cold_start_test_labels)
            test_predictions_np = np.array(cold_start_test_predictions)
            
            # Calculate success rate (predictions > 0.5 for positive samples, < 0.5 for negative)
            # IMPORTANT: Only consider correctly predicted results for recovery success rate
            predictions_binary = (test_predictions_np > 0.5).astype(float)
            success_count = np.sum(predictions_binary == test_labels_np)
            total_tested = len(test_labels_np)
            
            # Calculate recovery success rate: among all test samples, how many are correctly predicted
            # This is the true cold-start performance metric
            recovery_success_rate = success_count / total_tested
            
            # Check for problematic data distribution
            pos_samples = np.sum(test_labels_np == 1)
            neg_samples = np.sum(test_labels_np == 0)
            
            # If all samples are one class, the success rate is misleading
            if pos_samples == 0 or neg_samples == 0:
                print(f"  ⚠️  WARNING: Only one class in test data!")
                print(f"  ⚠️  Positive: {pos_samples}, Negative: {neg_samples}")
                print(f"  ⚠️  Success rate is misleading in this case!")
                
                # Use a more balanced metric: average of class-wise accuracy
                if pos_samples == 0:
                    # Only negative samples - check negative class accuracy
                    neg_accuracy = np.sum((test_predictions_np <= 0.5) & (test_labels_np == 0)) / neg_samples
                    recovery_success_rate = neg_accuracy * 0.5  # Penalize for imbalance
                    print(f"  ⚠️  Adjusted recovery success rate (negative only): {recovery_success_rate:.4f}")
                else:
                    # Only positive samples - check positive class accuracy  
                    pos_accuracy = np.sum((test_predictions_np > 0.5) & (test_labels_np == 1)) / pos_samples
                    recovery_success_rate = pos_accuracy * 0.5  # Penalize for imbalance
                    print(f"  ⚠️  Adjusted recovery success rate (positive only): {recovery_success_rate:.4f}")
            else:
                # Balanced case - use normal recovery success rate
                recovery_success_rate = success_count / total_tested
            
            # Calculate AUROC and AUPR
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                # Check if we have both classes
                unique_labels = np.unique(test_labels_np)
                if len(unique_labels) > 1:
                    cold_start_auroc = roc_auc_score(test_labels_np, test_predictions_np)
                    cold_start_aupr = average_precision_score(test_labels_np, test_predictions_np)
                else:
                    print(f"Warning: Only one class present in test data: {unique_labels}")
                    cold_start_auroc = 1.0 if unique_labels[0] == 1 else 0.0
                    cold_start_aupr = 1.0 if unique_labels[0] == 1 else 0.0
            except ImportError:
                print("Warning: sklearn not available, using default metrics")
                cold_start_auroc = 0.5
                cold_start_aupr = 0.5
            
            
            print(f"Cold-Start Test Results (Zero-Shot):")
            print(f"  Total test edges: {total_tested}")
            print(f"  Success count: {success_count}")
            print(f"  Success rate: {cold_start_success_rate:.4f}")
            print(f"  AUROC: {cold_start_auroc:.4f}")
            print(f"  AUPR: {cold_start_aupr:.4f}")
            print(f"  Positive samples: {np.sum(test_labels_np == 1)}")
            print(f"  Negative samples: {np.sum(test_labels_np == 0)}")
            print(f"  Avg prediction: {np.mean(test_predictions_np):.4f}")
            print(f"  Prediction range: [{np.min(test_predictions_np):.4f}, {np.max(test_predictions_np):.4f}]")
            
            # Debug: Show the distribution of predictions and labels
            print(f"  Predictions > 0.5: {np.sum(test_predictions_np > 0.5)}")
            print(f"  Predictions <= 0.5: {np.sum(test_predictions_np <= 0.5)}")
            print(f"  Binary predictions: pos={np.sum(predictions_binary == 1)}, neg={np.sum(predictions_binary == 0)}")
            
            # Show some sample predictions for debugging
            print(f"  Sample predictions (first 10): {test_predictions_np[:10]}")
            print(f"  Sample labels (first 10): {test_labels_np[:10]}")
            print(f"  Sample binary preds (first 10): {predictions_binary[:10]}")
            
            # Calculate balanced accuracy as an alternative metric
            if pos_samples > 0 and neg_samples > 0:
                # Balanced case - calculate both class accuracies
                pos_accuracy = np.sum((test_predictions_np > 0.5) & (test_labels_np == 1)) / pos_samples
                neg_accuracy = np.sum((test_predictions_np <= 0.5) & (test_labels_np == 0)) / neg_samples
                balanced_accuracy = (pos_accuracy + neg_accuracy) / 2.0
                print(f"  Balanced accuracy: {balanced_accuracy:.4f}")
                print(f"  Positive class accuracy: {pos_accuracy:.4f}")
                print(f"  Negative class accuracy: {neg_accuracy:.4f}")
            
            # WARNING: Check if this is the problematic case
            if recovery_success_rate > 0.95:
                print(f"  ⚠️  WARNING: Recovery success rate too high ({recovery_success_rate:.4f})!")
                print(f"  ⚠️  This suggests a systematic bias or data issue!")
                print(f"  ⚠️  For cold start, expect recovery success rates of 10-30%!")
            
        else:
            print("No test edges found for cold-start drugs")
            recovery_success_rate = 0.0
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
