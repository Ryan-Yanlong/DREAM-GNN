#!/usr/bin/env python3
"""
Gdataset cold-start experiment script - leakage-free version
Dedicated to evaluating a model trained on Gdataset.

Core functions:
1) Load trained drug and disease embeddings
2) Perform retrieval + aggregation in the trained joint space (avoid mixing spaces)
3) Support disease-conditional aggregation (conditional prototype / minimal cross-attention)
4) Evaluate different K values
"""

import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import scipy.io as sio
from tqdm import tqdm
import warnings
import time
import json
warnings.filterwarnings('ignore')


class GdatasetColdStart:
    """
    Gdataset cold-start experiment implementation.
    Leakage-free, using trained embeddings only where appropriate.
    """
    
    def __init__(self, device=None, disease_conditional_config=None):
        self.device = device if device else th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # Configuration for disease-conditional aggregation
        if disease_conditional_config is None:
            disease_conditional_config = {
                'alpha': 0.5,  # Weight for disease↔neighbor-drug similarity
                'beta': 0.5,   # Weight for query-drug↔neighbor-drug similarity
                'temperature': 0.07,  # Softmax temperature (smaller => sharper)
                'use_disease_aware': True
            }
        self.disease_conditional_config = disease_conditional_config
        
        print(f"Gdataset cold-start experiment - leakage-free")
        print(f"Device: {self.device}")
        print(f"Disease-conditional config: {disease_conditional_config}")
        
        # Load raw Gdataset data
        self.load_raw_data()
    
    def load_raw_data(self):
        """Load raw Gdataset data."""
        data_path = './raw_data/drug_data/Gdataset/Gdataset.mat'
        
        if not os.path.exists(data_path):
            print(f"Error: Gdataset data file not found: {data_path}")
            return False
        
        try:
            data = sio.loadmat(data_path)
            
            self.association_matrix = data['didr'].T  # Drug–disease association matrix
            self.drug_embed_raw = data['drug_embed']  # Raw drug embeddings
            self.disease_embed_raw = data['disease_embed']  # Raw disease embeddings
            
            self.num_drug = self.association_matrix.shape[0]
            self.num_disease = self.association_matrix.shape[1]
            
            print(f"✓ Gdataset loaded:")
            print(f"  #Drugs: {self.num_drug}")
            print(f"  #Diseases: {self.num_disease}")
            print(f"  #Associations: {np.sum(self.association_matrix == 1)}")
            print(f"  Positive rate: {np.sum(self.association_matrix == 1) / (self.num_drug * self.num_disease):.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error: Failed to load Gdataset: {e}")
            return False
    
    def load_fold_data(self, fold, model_dir='seed_experiments/seed_77'):
        """Load model and embeddings for a specific fold."""
        print(f"\nLoading fold {fold} ...")
        
        # Check file paths
        model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
        embeddings_path = os.path.join(model_dir, f"embeddings_fold{fold}.pth")
        metadata_path = os.path.join(model_dir, f"cold_start_metadata_fold{fold}.json")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False
            
        if not os.path.exists(embeddings_path):
            print(f"Error: Embeddings file not found: {embeddings_path}")
            return False
        
        try:
            # Load MLP decoder weights
            state_dict = th.load(model_path, map_location=self.device, weights_only=False)
            
            # Rebuild MLP decoder (consistent with training)
            self.mlp_decoder = nn.Sequential(
                nn.Linear(256, 128),  # 2 * 128 = 256 (drug_feat + disease_feat)
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)
            
            # Load weights
            self.mlp_decoder[0].weight.data = state_dict['decoder.lin1.weight'].to(self.device)
            self.mlp_decoder[0].bias.data = state_dict['decoder.lin1.bias'].to(self.device)
            self.mlp_decoder[2].weight.data = state_dict['decoder.lin2.weight'].to(self.device)
            self.mlp_decoder[2].bias.data = state_dict['decoder.lin2.bias'].to(self.device)
            self.mlp_decoder[4].weight.data = state_dict['decoder.lin3.weight'].to(self.device)
            self.mlp_decoder[4].bias.data = state_dict['decoder.lin3.bias'].to(self.device)
            
            self.mlp_decoder.eval()
            
            # Load embeddings
            embeddings_data = th.load(embeddings_path, map_location=self.device)
            
            # Important: ensure we only use drugs/diseases from the training set as neighbor candidates
            self.train_drug_indices = embeddings_data['train_drug_indices']
            self.train_disease_indices = embeddings_data['train_disease_indices']
            
            # Trained embeddings (train set); used as neighbor pool
            if 'drug_feats' in embeddings_data:
                self.drug_embeddings_trained = embeddings_data['drug_feats'].to(self.device)
                self.disease_embeddings_trained = embeddings_data['dis_feats'].to(self.device)
            else:
                self.drug_embeddings_trained = embeddings_data['drug_out'].to(self.device)
                self.disease_embeddings_trained = embeddings_data['dis_out'].to(self.device)

            # Full embeddings in the unified space (aligned with decoder) for retrieval and aggregation
            # Note: Vdrug/Vdisease are the final attention-fused representations, matching decoder dim (default 128)
            if 'Vdrug' in embeddings_data and 'Vdisease' in embeddings_data:
                self.drug_embeddings_all = embeddings_data['Vdrug'].to(self.device)
                self.disease_embeddings_all = embeddings_data['Vdisease'].to(self.device)
            else:
                # Fallback: if missing, use topology-based representations to stay in the same space as much as possible
                self.drug_embeddings_all = embeddings_data.get('drug_out', self.drug_embeddings_trained).to(self.device)
                self.disease_embeddings_all = embeddings_data.get('dis_out', self.disease_embeddings_trained).to(self.device)
            
            # Load metadata (if exists)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✓ Metadata loaded")
            else:
                self.metadata = {}
            
            print(f"✓ Fold data loaded")
            print(f"  #Train drugs: {len(self.train_drug_indices)}")
            print(f"  #Train diseases: {len(self.train_disease_indices)}")
            print(f"  Train drug embedding shape: {self.drug_embeddings_trained.shape}")
            print(f"  Train disease embedding shape: {self.disease_embeddings_trained.shape}")
            print(f"  All drug embedding shape: {self.drug_embeddings_all.shape}")
            print(f"  All disease embedding shape: {self.disease_embeddings_all.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error: Failed to load fold data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_trained_space_similarity(self, query_drug_idx, candidate_drug_indices):
        """Compute cosine similarity in the trained unified space (aligned with decoder)."""
        with th.no_grad():
            query_embed = self.drug_embeddings_all[query_drug_idx].unsqueeze(0)  # [1, D]
            candidate_embeds = self.drug_embeddings_all[candidate_drug_indices]  # [K, D]
            query_norm = F.normalize(query_embed, p=2, dim=1)  # [1, D]
            cand_norm = F.normalize(candidate_embeds, p=2, dim=1)  # [K, D]
            sims = th.mm(cand_norm, query_norm.t()).squeeze(1)  # [K]
        return sims.cpu().numpy()

    def compute_raw_similarity(self, query_drug_idx, candidate_drug_indices):
        """Compute similarity using raw drug_embed (leakage-free, only raw features for unseen drug)."""
        query_embed = self.drug_embed_raw[query_drug_idx]
        candidate_embeds = self.drug_embed_raw[candidate_drug_indices]
        similarities = []
        query_norm = np.linalg.norm(query_embed)
        for candidate_embed in candidate_embeds:
            candidate_norm = np.linalg.norm(candidate_embed)
            if query_norm > 0 and candidate_norm > 0:
                sim = float(np.dot(query_embed, candidate_embed) / (query_norm * candidate_norm))
            else:
                sim = 0.0
            similarities.append(sim)
        return np.array(similarities, dtype=np.float32)
    
    def compute_raw_similarity_disease(self, query_disease_idx, candidate_disease_indices):
        """Compute similarity using raw disease_embed (leakage-free, only raw features for unseen disease)."""
        query_embed = self.disease_embed_raw[query_disease_idx]
        candidate_embeds = self.disease_embed_raw[candidate_disease_indices]
        similarities = []
        query_norm = np.linalg.norm(query_embed)
        for candidate_embed in candidate_embeds:
            candidate_norm = np.linalg.norm(candidate_embed)
            if query_norm > 0 and candidate_norm > 0:
                sim = float(np.dot(query_embed, candidate_embed) / (query_norm * candidate_norm))
            else:
                sim = 0.0
            similarities.append(sim)
        return np.array(similarities, dtype=np.float32)
    
    def aggregate_drug_embeddings(self, query_drug_idx, candidate_drug_indices, k_values=[3, 5, 10, 15, 20]):
        """Leakage-safe aggregation for unseen drug:
        1) Use raw-space similarity for neighbor retrieval (strict cold-start; no trained query drug vector)
        2) Map selected training neighbors to the unified space and aggregate to form a pseudo-query (decoder space)
        """
        results = {}
        # 1) raw-space similarity for Top-K retrieval
        sims_raw = self.compute_raw_similarity(query_drug_idx, candidate_drug_indices)
        temp = self.disease_conditional_config.get('temperature', 0.07)

        for k in k_values:
            if k > len(candidate_drug_indices):
                continue
            
            # Select Top-K candidate drugs
            top_k_rank = np.argsort(-sims_raw)[:k]
            top_k_candidate_indices = candidate_drug_indices[top_k_rank]
            top_k_similarities_raw = sims_raw[top_k_rank]
            
            # Neighbor embeddings in unified space (aligned with decoder)
            top_k_embeddings = self.drug_embeddings_all[top_k_candidate_indices]
            
            # Compute weights (softmax over raw-sim)
            weights = F.softmax(th.tensor(top_k_similarities_raw, dtype=th.float32, device=self.device) / temp, dim=0)

            # Weighted aggregation to obtain pseudo-query (decoder space; leakage-free)
            aggregated_drug_embedding = th.sum(weights.unsqueeze(-1) * top_k_embeddings, dim=0)
            
            results[k] = {
                'aggregated_embedding': aggregated_drug_embedding,  # pseudo-query [D]
                'weights': weights.cpu().numpy(),
                'top_k_indices': top_k_candidate_indices,
                'top_k_similarities_raw': top_k_similarities_raw,
                'max_similarity': float(np.max(top_k_similarities_raw)),
                'avg_similarity': float(np.mean(top_k_similarities_raw))
            }
        return results

    def predict_associations_conditional(self, top_k_candidate_indices, top_k_similarities_raw):
        """
        Disease-conditional aggregation and prediction:
        1) Top-K neighbors are selected by raw-space similarity (done externally)
        2) Build a pseudo-query (decoder space): softmax over raw-sim to weight neighbor embeddings
        3) For each disease, combine (disease↔neighbor) and (pseudo↔neighbor) similarities to get weights
        4) Produce a disease-specific aggregated drug representation and decode with the disease embedding
        Returns probabilities for all diseases (numpy array, length=num_disease)
        """
        with th.no_grad():
            config = self.disease_conditional_config
            alpha = float(config.get('alpha', 0.7))
            beta = float(config.get('beta', 0.3))
            temp = float(config.get('temperature', 0.07))

            # Neighbor drug and disease embeddings
            neighbor_embeds = self.drug_embeddings_all[top_k_candidate_indices]  # [K, D]
            disease_embeds = self.disease_embeddings_all  # [M, D]

            # Normalize for cosine similarities
            neighbor_norm = F.normalize(neighbor_embeds, p=2, dim=1)  # [K, D]
            disease_norm = F.normalize(disease_embeds, p=2, dim=1)    # [M, D]

            # Pseudo-query (decoder space) based on raw-sim softmax weights
            w_raw = F.softmax(th.tensor(top_k_similarities_raw, dtype=th.float32, device=self.device) / temp, dim=0)  # [K]
            pseudo_query = th.sum(w_raw.unsqueeze(1) * neighbor_embeds, dim=0, keepdim=True)  # [1, D]
            pseudo_query_norm = F.normalize(pseudo_query, p=2, dim=1)  # [1, D]

            # Neighbor↔pseudo similarity: [K]
            sim_neighbor_pseudo = th.mm(neighbor_norm, pseudo_query_norm.t()).squeeze(1)  # [K]

            # Disease↔neighbor similarity: [M, K]
            sim_disease_neighbor = th.mm(disease_norm, neighbor_norm.t())  # [M, K]

            # Mixed similarity per disease: alpha * (disease↔neighbor) + beta * (pseudo↔neighbor)
            mixed_scores = alpha * sim_disease_neighbor + beta * sim_neighbor_pseudo.unsqueeze(0)

            # Softmax over neighbors dimension K
            attn_weights = F.softmax(mixed_scores / temp, dim=1)  # [M, K]

            # Generate a drug aggregation per disease: [M, K] @ [K, D] -> [M, D]
            aggregated_drug_per_disease = th.matmul(attn_weights, neighbor_embeds)

            # Concatenate with each disease embedding and decode
            combined = th.cat([aggregated_drug_per_disease, disease_embeds], dim=1)  # [M, 2D]
            logits = self.mlp_decoder(combined).squeeze(-1)  # [M]
            probs = th.sigmoid(logits).cpu().numpy()

        return probs
    
    def predict_associations(self, drug_embedding, disease_indices):
        """Predict drug–disease associations (consistent with training)."""
        with th.no_grad():
            # Get disease embeddings (unified space)
            disease_embeddings = self.disease_embeddings_all[disease_indices]
            
            # Expand drug embedding to match number of diseases
            drug_expanded = drug_embedding.unsqueeze(0).expand(len(disease_indices), -1)
            
            # Concatenate features (same as training)
            combined_features = th.cat([drug_expanded, disease_embeddings], dim=1)
            
            # MLP decoder
            logits = self.mlp_decoder(combined_features).squeeze(-1)
            
            # Apply sigmoid
            probabilities = th.sigmoid(logits).cpu().numpy()
        
        return probabilities
    
    def predict_associations_with_disease(self, disease_embedding, drug_indices):
        """Given a disease embedding, predict association probabilities with all drugs (consistent with training)."""
        with th.no_grad():
            drug_embeddings = self.drug_embeddings_all[drug_indices]
            disease_expanded = disease_embedding.unsqueeze(0).expand(len(drug_indices), -1)
            combined_features = th.cat([drug_embeddings, disease_expanded], dim=1)
            logits = self.mlp_decoder(combined_features).squeeze(-1)
            probabilities = th.sigmoid(logits).cpu().numpy()
        return probabilities

    def aggregate_disease_embeddings(self, query_disease_idx, candidate_disease_indices, k_values=[3, 5, 10, 15, 20]):
        """Leakage-safe aggregation for unseen disease:
        1) Use raw disease_embed for neighbor retrieval
        2) Map selected training neighbors to the unified space and aggregate to form a disease pseudo-query
        """
        results = {}
        sims_raw = self.compute_raw_similarity_disease(query_disease_idx, candidate_disease_indices)
        temp = self.disease_conditional_config.get('temperature', 0.07)

        for k in k_values:
            if k > len(candidate_disease_indices):
                continue

            top_k_rank = np.argsort(-sims_raw)[:k]
            top_k_candidate_indices = candidate_disease_indices[top_k_rank]
            top_k_similarities_raw = sims_raw[top_k_rank]

            top_k_embeddings = self.disease_embeddings_all[top_k_candidate_indices]

            weights = F.softmax(th.tensor(top_k_similarities_raw, dtype=th.float32, device=self.device) / temp, dim=0)
            aggregated_disease_embedding = th.sum(weights.unsqueeze(-1) * top_k_embeddings, dim=0)

            results[k] = {
                'aggregated_embedding': aggregated_disease_embedding,
                'weights': weights.cpu().numpy(),
                'top_k_indices': top_k_candidate_indices,
                'top_k_similarities_raw': top_k_similarities_raw,
                'max_similarity': float(np.max(top_k_similarities_raw)),
                'avg_similarity': float(np.mean(top_k_similarities_raw))
            }
        return results

    def predict_associations_conditional_disease(self, top_k_candidate_indices, top_k_similarities_raw):
        """Drug-conditional aggregation on the disease side: per-drug aggregated disease representation, then decode."""
        with th.no_grad():
            config = self.disease_conditional_config
            alpha = float(config.get('alpha', 0.7))
            beta = float(config.get('beta', 0.3))
            temp = float(config.get('temperature', 0.07))

            neighbor_embeds = self.disease_embeddings_all[top_k_candidate_indices]  # [K, D]
            drug_embeds = self.drug_embeddings_all  # [N, D]

            neighbor_norm = F.normalize(neighbor_embeds, p=2, dim=1)  # [K, D]
            drug_norm = F.normalize(drug_embeds, p=2, dim=1)          # [N, D]

            w_raw = F.softmax(th.tensor(top_k_similarities_raw, dtype=th.float32, device=self.device) / temp, dim=0)  # [K]
            pseudo_query = th.sum(w_raw.unsqueeze(1) * neighbor_embeds, dim=0, keepdim=True)  # [1, D]
            pseudo_query_norm = F.normalize(pseudo_query, p=2, dim=1)  # [1, D]

            sim_neighbor_pseudo = th.mm(neighbor_norm, pseudo_query_norm.t()).squeeze(1)  # [K]
            sim_drug_neighbor = th.mm(drug_norm, neighbor_norm.t())  # [N, K]

            mixed_scores = alpha * sim_drug_neighbor + beta * sim_neighbor_pseudo.unsqueeze(0)  # [N, K]
            attn_weights = F.softmax(mixed_scores / temp, dim=1)  # [N, K]

            aggregated_disease_per_drug = th.matmul(attn_weights, neighbor_embeds)  # [N, D]

            combined = th.cat([drug_embeds, aggregated_disease_per_drug], dim=1)  # [N, 2D]
            logits = self.mlp_decoder(combined).squeeze(-1)  # [N]
            probs = th.sigmoid(logits).cpu().numpy()

        return probs
    
    def evaluate_cold_start(self, num_test_drugs=50, k_values=[3, 5, 10, 15, 20], random_seed=42):
        """Evaluate cold-start performance (unseen drug)."""
        np.random.seed(random_seed)
        
        print(f"\n=== Cold-start evaluation (unseen drug) ===")
        print(f"#Test drugs: {num_test_drugs}")
        print(f"K values: {k_values}")
        
        # Randomly select test drugs
        all_drug_indices = np.arange(self.num_drug)
        test_drug_indices = np.random.choice(all_drug_indices, num_test_drugs, replace=False)
        
        # Ensure candidates come from the training set
        candidate_drug_indices = np.setdiff1d(self.train_drug_indices, test_drug_indices)
        
        print(f"  #Test drugs: {len(test_drug_indices)}")
        print(f"  #Candidate drugs: {len(candidate_drug_indices)}")
        
        # Container for different K values
        k_results = {k: {'predictions': [], 'labels': [], 'drug_details': []} for k in k_values}
        
        use_disease_aware = bool(self.disease_conditional_config.get('use_disease_aware', True))

        for drug_idx in tqdm(test_drug_indices, desc="Evaluating drugs"):
            # Ground-truth labels
            true_associations = self.association_matrix[drug_idx]
            true_positive_diseases = np.where(true_associations == 1)[0]
            num_positives = len(true_positive_diseases)
            
            if num_positives == 0:
                continue
            
            # Neighbor retrieval in unified space and (unconditional) aggregation for stats/baseline
            aggregation_results = self.aggregate_drug_embeddings(drug_idx, candidate_drug_indices, k_values)

            # Predict for each K
            for k, agg_result in aggregation_results.items():
                if use_disease_aware:
                    # Disease-conditional aggregation + decoding (leakage-free: no trained query drug vector)
                    predictions = self.predict_associations_conditional(
                        agg_result['top_k_indices'], agg_result['top_k_similarities_raw']
                    )
                else:
                    # Unconditional aggregation + decoding (baseline)
                    predictions = self.predict_associations(
                        agg_result['aggregated_embedding'], np.arange(self.num_disease)
                    )

                # Record results
                k_results[k]['predictions'].extend(predictions)
                k_results[k]['labels'].extend(true_associations)

                # Per-drug analysis details
                drug_detail = {
                    'drug_idx': drug_idx,
                    'num_positives': num_positives,
                    'max_similarity': agg_result['max_similarity'],
                    'avg_similarity': agg_result['avg_similarity'],
                    'k_used': k,
                    'positive_scores': predictions[true_positive_diseases]
                }
                k_results[k]['drug_details'].append(drug_detail)
        
        # Compute metrics for each K
        final_results = {}
        for k in k_values:
            if len(k_results[k]['predictions']) > 0:
                predictions = np.array(k_results[k]['predictions'])
                labels = np.array(k_results[k]['labels'])
                
                # Metrics
                auroc = roc_auc_score(labels, predictions)
                aupr = average_precision_score(labels, predictions)
                
                # Threshold metrics
                threshold_metrics = {}
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    y_pred = (predictions >= threshold).astype(int)
                    precision = precision_score(labels, y_pred, zero_division=0)
                    recall = recall_score(labels, y_pred, zero_division=0)
                    f1 = f1_score(labels, y_pred, zero_division=0)
                    
                    threshold_metrics[threshold] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                
                final_results[k] = {
                    'auroc': auroc,
                    'aupr': aupr,
                    'threshold_metrics': threshold_metrics,
                    'drug_details': k_results[k]['drug_details']
                }
        
        return final_results

    def evaluate_cold_start_unseen_disease(self, num_test_diseases=50, k_values=[3, 5, 10, 15, 20], random_seed=42):
        """Evaluate cold-start performance (unseen disease; symmetric implementation)."""
        np.random.seed(random_seed)

        print(f"\n=== Cold-start evaluation (unseen disease) ===")
        print(f"#Test diseases: {num_test_diseases}")
        print(f"K values: {k_values}")

        all_disease_indices = np.arange(self.num_disease)
        test_disease_indices = np.random.choice(all_disease_indices, num_test_diseases, replace=False)

        candidate_disease_indices = np.setdiff1d(self.train_disease_indices, test_disease_indices)

        print(f"  #Test diseases: {len(test_disease_indices)}")
        print(f"  #Candidate diseases: {len(candidate_disease_indices)}")

        k_results = {k: {'predictions': [], 'labels': [], 'disease_details': []} for k in k_values}

        use_conditional = bool(self.disease_conditional_config.get('use_disease_aware', True))

        for disease_idx in tqdm(test_disease_indices, desc="Evaluating diseases"):
            true_associations = self.association_matrix[:, disease_idx]
            pos_drugs = np.where(true_associations == 1)[0]
            num_positives = len(pos_drugs)

            if num_positives == 0:
                continue

            aggregation_results = self.aggregate_disease_embeddings(disease_idx, candidate_disease_indices, k_values)

            for k, agg_result in aggregation_results.items():
                if use_conditional:
                    predictions = self.predict_associations_conditional_disease(
                        agg_result['top_k_indices'], agg_result['top_k_similarities_raw']
                    )
                else:
                    predictions = self.predict_associations_with_disease(
                        agg_result['aggregated_embedding'], np.arange(self.num_drug)
                    )

                k_results[k]['predictions'].extend(predictions)
                k_results[k]['labels'].extend(true_associations)

                disease_detail = {
                    'disease_idx': disease_idx,
                    'num_positives': num_positives,
                    'max_similarity': agg_result['max_similarity'],
                    'avg_similarity': agg_result['avg_similarity'],
                    'k_used': k,
                    'positive_scores': predictions[pos_drugs]
                }
                k_results[k]['disease_details'].append(disease_detail)

        final_results = {}
        for k in k_values:
            if len(k_results[k]['predictions']) > 0:
                predictions = np.array(k_results[k]['predictions'])
                labels = np.array(k_results[k]['labels'])

                auroc = roc_auc_score(labels, predictions)
                aupr = average_precision_score(labels, predictions)

                threshold_metrics = {}
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    y_pred = (predictions >= threshold).astype(int)
                    precision = precision_score(labels, y_pred, zero_division=0)
                    recall = recall_score(labels, y_pred, zero_division=0)
                    f1 = f1_score(labels, y_pred, zero_division=0)

                    threshold_metrics[threshold] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

                final_results[k] = {
                    'auroc': auroc,
                    'aupr': aupr,
                    'threshold_metrics': threshold_metrics,
                    'disease_details': k_results[k]['disease_details']
                }

        return final_results
    
    def analyze_results(self, results):
        """Analyze results."""
        print(f"\n=== Results analysis ===")
        
        # 1) Overall performance across K
        print(f"Performance across K:")
        print(f"{'K':<6} {'AUROC':<8} {'AUPR':<8} {'F1@0.3':<12}")
        print("-" * 40)
        
        for k in sorted(results.keys()):
            auroc = results[k]['auroc']
            aupr = results[k]['aupr']
            f1_03 = results[k]['threshold_metrics'][0.3]['f1']
            print(f"{k:<6} {auroc:<8.4f} {aupr:<8.4f} {f1_03:<12.4f}")
        
        # 2) Similarity analysis
        print(f"\nSimilarity analysis:")
        for k in sorted(results.keys()):
            details = results[k].get('drug_details') or results[k].get('disease_details') or []
            max_sims = [detail['max_similarity'] for detail in details]
            avg_sims = [detail['avg_similarity'] for detail in details]
            
            print(f"K={k}: Max similarity={np.mean(max_sims):.4f}±{np.std(max_sims):.4f}, "
                  f"Avg similarity={np.mean(avg_sims):.4f}±{np.std(avg_sims):.4f}")
        
        # 3) Positive recovery analysis
        print(f"\nPositive recovery (threshold=0.3):")
        for k in sorted(results.keys()):
            details = results[k].get('drug_details') or results[k].get('disease_details') or []
            recovery_rates = []
            
            for detail in details:
                positive_scores = detail['positive_scores']
                recovered = np.sum(positive_scores >= 0.3)
                recovery_rate = recovered / detail['num_positives']
                recovery_rates.append(recovery_rate)
            
            if recovery_rates:
                avg_recovery = np.mean(recovery_rates)
                print(f"K={k}: Avg recovery={avg_recovery:.4f} (n={len(recovery_rates)})")
    
    def run_experiment(self, cv_folds=3, model_dir='seed_experiments/seed_77', unseen_mode='drug'):
        """Run the full experiment across folds."""
        all_results = []
        
        for fold in range(1, cv_folds + 1):
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"{'='*60}")
            
            try:
                if not self.load_fold_data(fold, model_dir):
                    continue
                
                # Run cold-start evaluation
                if unseen_mode == 'disease':
                    results = self.evaluate_cold_start_unseen_disease(
                        num_test_diseases=30,
                        k_values=[3, 5, 10, 15, 20],
                        random_seed=77 + fold
                    )
                else:
                    results = self.evaluate_cold_start(
                        num_test_drugs=50,
                        k_values=[3, 5, 10, 15, 20],
                        random_seed=77 + fold
                    )
                
                # Analyze
                self.analyze_results(results)
                all_results.append(results)
                
            except Exception as e:
                print(f"Fold {fold} failed: {e}")
                import traceback
                traceback.print_exc()
        
        return all_results


def main():
    """Entry point."""
    print(f"Gdataset cold-start experiment")
    print(f"{'='*80}")
    
    # Initialize cold-start experiment
    cold_start = GdatasetColdStart()
    
    # Check data loaded
    if not hasattr(cold_start, 'association_matrix'):
        print("Error: Data loading failed. Please check Gdataset.mat")
        return
    
    # Run experiment
    print(f"\nStart running cold-start experiments...")
    results = cold_start.run_experiment(
        cv_folds=3,
        model_dir='seed_experiments/seed_77',
        unseen_mode='drug'  # Set to 'disease' to evaluate unseen disease cold-start
    )
    
    print(f"\nFinished!")
    print(f"Completed {len(results)} folds")


if __name__ == "__main__":
    main()
