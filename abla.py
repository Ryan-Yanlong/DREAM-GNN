# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
import copy
import traceback

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# =======================================================================
#                           Dependency Imports
# =======================================================================
# Assuming these imports exist and work from your project structure
# Make sure these files are in the same directory or accessible in Python path.
try:
    from model import Net
    from data import DrugDataLoader
    from utils import setup_seed, MetricLogger, common_loss
    # Import evaluate function (adjust path if needed)
    try:
        from evaluate import evaluate
    except ImportError:
        print("Warning: Could not import 'evaluate' function from evaluate.py.")
        # Define a dummy evaluate if needed for the code to run
        def evaluate(*args, **kwargs):
            print("Warning: Using dummy evaluate function!")
            return 0.5, 0.5 # Dummy AUROC, AUPR

    # Import graph augmentation function (adjust path if needed)
    try:
        from graph_augmentation import augment_graph_data
    except ImportError:
         print("Warning: Could not import 'augment_graph_data' from graph_augmentation.py.")
         # Define a dummy augmentation if needed for the code to run
         def augment_graph_data(graph_data, *args, **kwargs):
             print("Warning: Using dummy graph augmentation!")
             return graph_data # Return original data

except ImportError as e:
    print(f"Error importing project modules (model, data, utils, etc.): {e}")
    print("Please ensure necessary Python files are accessible.")
    exit()

# =======================================================================
#                       Helper Functions & Classes
# =======================================================================

class LabelSmoothingBCELoss(nn.Module):
    """ Custom BCE Loss with Label Smoothing """
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        print(f"Initialized LabelSmoothingBCELoss with smoothing={smoothing}")

    def forward(self, pred, target):
        if self.smoothing > 0.0:
            with th.no_grad(): # Don't track gradients for target manipulation
                smooth_target = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        else:
            smooth_target = target.float()
        return F.binary_cross_entropy_with_logits(pred, smooth_target)


def get_top_novel_predictions(args, model, dataset, cv_idx, top_k=200):
    """
    Gets top K predictions (drug-disease pairs) that are not present in the ground truth.
    NOTE: This function is defined but not called by default in the ablation setup.
          Enable via --allow-generate_top_predictions flag.
    """
    print(f"Attempting generation of top {top_k} novel predictions for fold {cv_idx+1}...")
    model.eval()
    current_device = args.device

    # --- Safely access dataset attributes ---
    try:
        ground_truth = dataset.association_matrix
        num_drugs = dataset.num_drug
        num_diseases = dataset.num_disease
        # Ensure dataset object has methods/attributes needed below
        if not all(hasattr(dataset, attr) for attr in ['data_cv', 'cv_specific_graphs',
                                'drug_sim_features', 'disease_sim_features', 'drug_feature',
                                'disease_feature', '_generate_dec_graph']):
             print("Error: Dataset object missing required attributes/methods for prediction generation.")
             return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])
    except AttributeError as e:
        print(f"Error accessing required dataset attributes: {e}")
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])

    # --- Get graph data for prediction ---
    # Use non-augmented graphs/features corresponding to the state used for final evaluation
    try:
        cv_data = dataset.data_cv[cv_idx]
        fold_specific_graphs = dataset.cv_specific_graphs[cv_idx] # Graphs used for this fold's test set
        drug_graph = fold_specific_graphs['drug_graph'].to(current_device)
        dis_graph = fold_specific_graphs['disease_graph'].to(current_device)
        drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(current_device)
        disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(current_device)
        drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(current_device)
        dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(current_device)
        drug_feat = dataset.drug_feature.to(current_device)
        dis_feat = dataset.disease_feature.to(current_device)
        # Use the training graph structure for the final node embeddings learned
        enc_graph = cv_data['train'][0].int().to(current_device)
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        print(f"Error accessing/processing graph/feature data for prediction generation: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])
    except Exception as e: # Catch-all for other unexpected errors
        print(f"Unexpected error preparing data for prediction: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])

    # --- Create novel pairs ---
    novel_pairs = []
    if isinstance(ground_truth, np.ndarray):
        gt_shape = ground_truth.shape
        for r in range(num_drugs):
            for c in range(num_diseases):
                 # Check if indices are within bounds before accessing
                 if r < gt_shape[0] and c < gt_shape[1]:
                     if ground_truth[r, c] == 0:
                         novel_pairs.append((r, c))
                 else:
                     print(f"Warning: Skipping index check ({r}, {c}) outside ground_truth shape {gt_shape}")
    else:
        print("Error: Ground truth association matrix is not a NumPy array.")
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])

    print(f"Found {len(novel_pairs)} potential novel drug-disease pairs.")
    if not novel_pairs: return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])

    # --- Batch Prediction ---
    batch_size = getattr(args, 'prediction_batch_size', 5000) # Allow configurable batch size
    num_batches = len(novel_pairs) // batch_size + (1 if len(novel_pairs) % batch_size > 0 else 0)
    all_predictions = []
    print(f"Predicting scores in {num_batches} batches...")

    with th.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(novel_pairs))
            batch_pairs_list = novel_pairs[start_idx:end_idx]
            if not batch_pairs_list: continue

            try:
                rating_pairs = (
                    np.array([p[0] for p in batch_pairs_list], dtype=np.int64),
                    np.array([p[1] for p in batch_pairs_list], dtype=np.int64)
                )
                if not rating_pairs[0].size: continue

                dec_graph = dataset._generate_dec_graph(rating_pairs).to(current_device)

                # Ensure model gets all necessary arguments
                pred_ratings, *_ = model( # Use *_ to ignore unused outputs
                    enc_graph, dec_graph,
                    drug_graph, drug_sim_feat, drug_feat,
                    dis_graph, dis_sim_feat, dis_feat,
                    drug_feature_graph, disease_feature_graph
                    # Pass Two_Stage=False if necessary
                )

                pred_scores = th.sigmoid(pred_ratings.squeeze(-1)).cpu().numpy()

                for j, (drug_id, disease_id) in enumerate(batch_pairs_list):
                    if j < len(pred_scores):
                        all_predictions.append({
                            'drug_id': drug_id, 'disease_id': disease_id, 'score': float(pred_scores[j])
                        })

                if (i + 1) % max(1, num_batches // 10) == 0 or (i + 1) == num_batches:
                    print(f"  Processed prediction batch {i+1}/{num_batches}")

            except AttributeError as ae:
                if '_generate_dec_graph' in str(ae): print(f"Error: Dataset object missing '_generate_dec_graph' method.")
                else: print(f"AttributeError processing prediction batch {i+1}/{num_batches}: {str(ae)}")
                traceback.print_exc(); break # Stop if critical method missing
            except Exception as e:
                print(f"Error processing prediction batch {i+1}/{num_batches}: {str(e)}")
                traceback.print_exc(); continue # Skip batch on error

    if not all_predictions:
        print("Warning: No predictions were generated after batch processing.")
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])

    # --- Process and Save Results ---
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)

    if hasattr(dataset, 'drug_ids') and dataset.drug_ids is not None:
        # Simplified name mapping
        pred_df['drug_name'] = pred_df['drug_id'].apply(lambda x: f"Drug_{x}")

    top_pred_df = pred_df.head(top_k)

    try:
        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        csv_path = os.path.join(args.save_dir, f"top{top_k}_novel_predictions_fold{cv_idx+1}.csv")
        top_pred_df.to_csv(csv_path, index=False)
        print(f"Top {top_k} novel predictions saved to {csv_path}")
    except Exception as e: print(f"Error saving predictions CSV: {str(e)}")

    return top_pred_df


def train(args, dataset, cv):
    """ Model training process for one CV fold. """
    print(f"Starting training for fold {cv+1}...")
    current_device = args.device # Device string

    # --- Set dimensions and rating values ---
    try:
        args.src_in_units = dataset.drug_feature_shape[1]
        args.dst_in_units = dataset.disease_feature_shape[1]
        args.fdim_drug = dataset.drug_feature_shape[0]
        args.fdim_disease = dataset.disease_feature_shape[0]
    except AttributeError as e: print(f"Error accessing feature shapes: {e}"); return np.nan, np.nan
    try:
         # Ensure ratings are converted to list of unique ints/floats
         raw_ratings = dataset.cv_data_dict[cv][2].numpy()
         args.rating_vals = sorted(list(np.unique(raw_ratings).astype(raw_ratings.dtype)))
         print(f"[Model] Using rating values from dataset fold {cv+1}: {args.rating_vals}")
    except (AttributeError, KeyError, IndexError, TypeError) as e:
         print(f"Error accessing rating values: {e}. Using default [0, 1]."); args.rating_vals = [0, 1]

    # --- Prepare data for the fold ---
    try:
        cv_data = dataset.data_cv[cv]
        if hasattr(dataset, 'cv_specific_graphs') and cv is not None and cv in dataset.cv_specific_graphs:
             fold_specific_graphs = dataset.cv_specific_graphs[cv]
        else: fold_specific_graphs = {'drug_graph': dataset.drug_graph, 'disease_graph': dataset.disease_graph, 'drug_feature_graph': dataset.drug_feature_graph, 'disease_feature_graph': dataset.disease_feature_graph}

        drug_graph = fold_specific_graphs['drug_graph'].to(current_device)
        dis_graph = fold_specific_graphs['disease_graph'].to(current_device)
        drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(current_device)
        disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(current_device)
        drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(current_device)
        dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(current_device)
        drug_feat = dataset.drug_feature.to(current_device)
        dis_feat = dataset.disease_feature.to(current_device)
        train_enc_graph = cv_data['train'][0].int().to(current_device)
        train_dec_graph = cv_data['train'][1].int().to(current_device)
        train_gt_ratings = cv_data['train'][2].to(current_device)
        train_data_dict = {'test': cv_data['train']}
        test_data_dict = {'test': cv_data['test']}
    except Exception as e: print(f"Error preparing data for CV fold {cv+1}: {e}"); traceback.print_exc(); return np.nan, np.nan

    # --- Initialize Model, Loss, Optimizer ---
    try: model = Net(args=args).to(current_device)
    except Exception as e: print(f"Error initializing Net model: {e}"); traceback.print_exc(); return np.nan, np.nan

    rel_loss_fn = LabelSmoothingBCELoss(smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=getattr(args, 'lr_scheduler_patience', 10), factor=0.5, verbose=True)
    print(f"Using loss: {type(rel_loss_fn).__name__}")
    print("Optimizer and Scheduler initialized.")

    # --- Logger ---
    log_dir = args.save_dir
    test_loss_logger = MetricLogger(['iter', 'loss', 'train_auroc', 'train_aupr', 'test_auroc', 'test_aupr'],
                                  ['%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
                                  os.path.join(log_dir, f'test_metric_fold{args.save_id}.csv'))
    print(f"Logging metrics to: {log_dir}")

    # --- Augmentation Setup ---
    use_aug = args.use_augmentation
    aug_methods_list = args.aug_methods if use_aug else []
    aug_params = { k: getattr(args, k, 0.0) for k in ['edge_dropout_rate', 'feature_noise_scale', 'graph_noise_scale', 'add_edge_rate', 'feature_mask_rate', 'mixup_alpha']}
    if use_aug: print(f"Using Augmentations: {aug_methods_list}")
    else: print("Data Augmentation Disabled.")

    # --- Training Loop ---
    print("Starting training loop...")
    best_aupr = -1.0; best_auroc = 0.0; best_iter = 0; best_train_aupr = 0.0; best_train_auroc = 0.0
    start_time = time.perf_counter()

    for iter_idx in range(1, args.train_max_iter + 1):
        model.train()
        # --- Augmentation ---
        if use_aug:
            graph_data_to_augment = {
                'enc_graph': train_enc_graph, 'drug_graph': drug_graph, 'disease_graph': dis_graph,
                'drug_feature_graph': drug_feature_graph, 'disease_feature_graph': disease_feature_graph,
                'drug_feat': drug_feat, 'disease_feat': dis_feat,
                'drug_sim_feat': drug_sim_feat, 'disease_sim_feat': dis_sim_feat
            }
            try: augmented_data = augment_graph_data(graph_data_to_augment, aug_methods_list, aug_params)
            except Exception as e: print(f"Warning: Augmentation error at iter {iter_idx}: {e}"); augmented_data = graph_data_to_augment # Fallback
            iter_train_enc_graph, iter_train_dec_graph = augmented_data.get('enc_graph',train_enc_graph), train_dec_graph
            iter_drug_graph, iter_dis_graph = augmented_data.get('drug_graph',drug_graph), augmented_data.get('disease_graph',dis_graph)
            iter_drug_feature_graph, iter_disease_feature_graph = augmented_data.get('drug_feature_graph',drug_feature_graph), augmented_data.get('disease_feature_graph',disease_feature_graph)
            iter_drug_feat, iter_dis_feat = augmented_data.get('drug_feat',drug_feat), augmented_data.get('disease_feat',dis_feat)
            iter_drug_sim_feat, iter_dis_sim_feat = augmented_data.get('drug_sim_feat',drug_sim_feat), augmented_data.get('disease_sim_feat',dis_sim_feat)
        else:
            iter_train_enc_graph, iter_train_dec_graph = train_enc_graph, train_dec_graph; iter_drug_graph, iter_dis_graph = drug_graph, dis_graph
            iter_drug_feature_graph, iter_disease_feature_graph = drug_feature_graph, disease_feature_graph; iter_drug_feat, iter_dis_feat = drug_feat, dis_feat
            iter_drug_sim_feat, iter_dis_sim_feat = drug_sim_feat, dis_sim_feat

        # --- Forward Pass ---
        try:
            # Pass all required graph/feature arguments to the model
            pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, *_ = model(
                iter_train_enc_graph, iter_train_dec_graph,
                iter_drug_graph, iter_drug_sim_feat, iter_drug_feat,
                iter_dis_graph, iter_dis_sim_feat, iter_dis_feat,
                iter_drug_feature_graph, iter_disease_feature_graph
            )
            pred_ratings = pred_ratings.squeeze(-1)
        except Exception as e: print(f"Error forward pass iter {iter_idx}: {e}"); traceback.print_exc(); continue

        # --- Loss Calculation ---
        try:
            loss_com_drug = common_loss(drug_out, drug_sim_out) if args.beta > 0 and drug_out is not None and drug_sim_out is not None else 0.0
            loss_com_dis = common_loss(dis_out, dis_sim_out) if args.beta > 0 and dis_out is not None and dis_sim_out is not None else 0.0
            rel_loss_val = rel_loss_fn(pred_ratings, train_gt_ratings)
            total_loss = rel_loss_val + args.beta * (loss_com_drug + loss_com_dis)
            # Apply L2 regularization if weight > 0
            if args.l2_reg_weight > 0:
                l2_penalty = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad) / 2.0
                total_loss += args.l2_reg_weight * l2_penalty
        except Exception as e: print(f"Error loss calc iter {iter_idx}: {e}"); continue

        # --- Backward Pass & Optimization ---
        optimizer.zero_grad()
        try:
            total_loss.backward()
            if args.train_grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
            optimizer.step()
        except Exception as e: print(f"Error backward/step iter {iter_idx}: {e}")

        # --- Validation ---
        if iter_idx % args.train_valid_interval == 0:
            model.eval()
            with th.no_grad():
                try:
                    # Pass necessary non-augmented graphs/features for consistent evaluation
                    train_auroc, train_aupr = evaluate(args, model, train_data_dict,
                                                       drug_graph, drug_feat, drug_sim_feat,
                                                       dis_graph, dis_feat, dis_sim_feat,
                                                       drug_feature_graph, disease_feature_graph)
                    test_auroc, test_aupr = evaluate(args, model, test_data_dict,
                                                     drug_graph, drug_feat, drug_sim_feat,
                                                     dis_graph, dis_feat, dis_sim_feat,
                                                     drug_feature_graph, disease_feature_graph)
                except Exception as e: print(f"Error evaluation iter {iter_idx}: {e}"); train_auroc, train_aupr, test_auroc, test_aupr = np.nan, np.nan, np.nan, np.nan

            if not np.isnan(test_aupr): scheduler.step(test_aupr) # Step scheduler based on validation AUPR
            if not (np.isnan(train_auroc) or np.isnan(test_auroc)):
                test_loss_logger.log(iter=iter_idx, loss=total_loss.item(), train_auroc=train_auroc, train_aupr=train_aupr, test_auroc=test_auroc, test_aupr=test_aupr)
                print(f"Iter={iter_idx:5d}, Loss={total_loss.item():.4f}, LR={optimizer.param_groups[0]['lr']:.1E}, Train AUROC={train_auroc:.4f}, AUPR={train_aupr:.4f}, Test AUROC={test_auroc:.4f}, AUPR={test_aupr:.4f}")
                # Update best metrics based on test AUPR
                if test_aupr > best_aupr:
                    best_aupr, best_auroc, best_train_aupr, best_train_auroc, best_iter = test_aupr, test_auroc, train_aupr, train_auroc, iter_idx
                    print(f"*** Best Test AUPR {best_aupr:.4f} at iter {best_iter} ***")
                    if args.save_model:
                        model_path = os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")
                        try: th.save(model.state_dict(), model_path); print(f"Saved best model to {model_path}")
                        except Exception as e: print(f"!!! Error saving model: {e}")
            else: print(f"Iter={iter_idx:5d}, Evaluation failed or produced NaN.")

    # --- End of Training ---
    elapsed_time = time.perf_counter() - start_time; test_loss_logger.close()
    print("Training finished. Run time:", time.strftime("%H:%M:%S", time.gmtime(round(elapsed_time))))
    if best_iter > 0:
        print(f"Best Iter Fold {args.save_id}: {best_iter} | Test AUROC={best_auroc:.4f}, AUPR={best_aupr:.4f}")
        best_metrics_path = os.path.join(args.save_dir, f"best_metric_fold{args.save_id}.csv")
        try:
            pd.DataFrame([{'iter': best_iter, 'train_auroc': best_train_auroc, 'train_aupr': best_train_aupr, 'test_auroc': best_auroc, 'test_aupr': best_aupr}]).to_csv(best_metrics_path, index=False)
            print(f"Best metrics saved to {best_metrics_path}")
        except Exception as e: print(f"!!! Error saving best metrics: {e}")
    else: print(f"Fold {args.save_id}: No valid best model found."); best_auroc, best_aupr = np.nan, np.nan

    # --- Post-Training Predictions (if enabled) ---
    if args.generate_top_predictions and args.save_model and best_iter > 0:
        print("\nGenerating novel predictions post-training...")
        best_model_path = os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")
        if os.path.exists(best_model_path):
            try:
                # Re-initialize the model structure with the same args used for training
                best_model = Net(args=args).to(current_device)
                best_model.load_state_dict(th.load(best_model_path, map_location=current_device))
                _ = get_top_novel_predictions(args, best_model, dataset, cv, top_k=args.top_k)
            except Exception as e: print(f"!!! Error prediction generation: {e}"); traceback.print_exc()
        else: print("Best model file not found, cannot generate predictions.")

    return best_auroc, best_aupr


# =======================================================================
#                  Ablation Experiment Runner Function
# =======================================================================
def run_experiment(args):
    """ Runs one ablation experiment configuration using 10-fold CV. """
    start_time = time.time()
    print(f"\n===== Running Experiment with Config =====")
    # Print config details efficiently
    config_str = " | ".join([f"{k}: {v}" for k, v in sorted(vars(args).items())])
    print(f"Config: {config_str[:200]}...") # Print start of config
    print(f"Results will be saved in: {args.save_dir}")

    # --- Setup ---
    seed = 0 # Use a fixed seed for reproducibility across experiments
    setup_seed(seed)
    print(f"Using fixed Random Seed for comparability: {seed}")

    # Ensure save directory exists
    base_dir = os.path.dirname(args.save_dir)
    if base_dir and not os.path.isdir(base_dir): os.makedirs(base_dir, exist_ok=True)
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir, exist_ok=True)

    # Set device string in args (used by train and others)
    if isinstance(args.device, int) and args.device >= 0 and th.cuda.is_available():
        device_str = f"cuda:{args.device}"
        th.cuda.set_device(args.device)
    else: device_str = "cpu"
    args.device = device_str # Overwrite args.device with the string ('cuda:0' or 'cpu')
    print(f"Using device: {args.device}")

    # --- Data Loading ---
    print(f"Loading dataset: {args.data_name}...")
    try:
        # Pass necessary params to DrugDataLoader
        current_aug_params = { k: getattr(args, k, 0.0) for k in ['edge_dropout_rate', 'feature_noise_scale', 'graph_noise_scale', 'add_edge_rate', 'feature_mask_rate']}
        dataset = DrugDataLoader(args.data_name, args.device, symm=args.gcn_agg_norm_symm, k=args.num_neighbor,
                                 use_augmentation=args.use_augmentation, aug_params=current_aug_params)
        # Set embedding mode if the dataloader uses it
        if hasattr(dataset, 'embedding_mode'):
             dataset.embedding_mode = args.embedding_mode
        print("Dataset loaded successfully.")
    except Exception as e: print(f"!!! FATAL: Error loading dataset {args.data_name}: {e}"); traceback.print_exc(); return np.nan, np.nan, np.nan, np.nan

    # --- Cross-Validation ---
    all_fold_metrics = []
    num_folds = 10 # Assuming 10-fold CV
    print(f"Starting {num_folds}-fold cross-validation...")
    for cv in range(num_folds):
        fold_start_time = time.time()
        args.save_id = cv + 1 # For logging/saving within the fold
        print(f"\n============== Fold {cv + 1}/{num_folds} ==============")
        try:
            # Call the main train function for this fold
            auroc, aupr = train(args, dataset, cv)
            all_fold_metrics.append({'fold': cv+1, 'auroc': auroc, 'aupr': aupr})
        except Exception as e:
             print(f"!!! Error executing train function for fold {cv+1}: {e}")
             traceback.print_exc()
             all_fold_metrics.append({'fold': cv+1, 'auroc': np.nan, 'aupr': np.nan})
        fold_duration = time.time() - fold_start_time
        print(f"Fold {cv+1} duration: {fold_duration:.2f} seconds")

    # --- Aggregate Results ---
    # Filter out NaN values before calculating mean/std
    valid_aurocs = [m['auroc'] for m in all_fold_metrics if pd.notna(m['auroc'])]
    valid_auprs = [m['aupr'] for m in all_fold_metrics if pd.notna(m['aupr'])]
    avg_auroc = np.mean(valid_aurocs) if valid_aurocs else np.nan
    avg_aupr = np.mean(valid_auprs) if valid_auprs else np.nan
    std_auroc = np.std(valid_aurocs) if len(valid_aurocs) > 1 else 0.0
    std_aupr = np.std(valid_auprs) if len(valid_auprs) > 1 else 0.0
    num_successful_folds = len(valid_aurocs)

    print(f"\n===== Experiment Aggregated Results ({num_successful_folds}/{num_folds} successful folds) =====")
    print(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Average AUPR: {avg_aupr:.4f} ± {std_aupr:.4f}")

    # Save fold metrics summary for this experiment
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_path = os.path.join(args.save_dir, "fold_metrics_summary.csv")
    try: metrics_df.to_csv(metrics_path, index=False); print(f"Fold metrics summary saved to {metrics_path}")
    except Exception as e: print(f"!!! Error saving fold metrics summary: {e}")

    end_time = time.time()
    print(f"Total experiment duration: {end_time - start_time:.2f} seconds")
    return avg_auroc, avg_aupr, std_auroc, std_aupr

# =======================================================================
#                           Main Execution Block
# =======================================================================
if __name__ == '__main__':
    # --- 使用与之前脚本相同的参数解析器设置 ---
    parser = argparse.ArgumentParser(description='Ablation Study V4 (AdaDR structure)')
    # Basic Setup
    parser.add_argument('--device', default='0', type=int, help='GPU device ID, -1 for CPU')
    # !!! 修改保存目录 !!!
    parser.add_argument('--base_save_dir', type=str, default='ablation_study_v4_results', help='Base directory for V4 results')
    parser.add_argument('--data_name', default='lrssl', type=str, help='Name of the dataset') # 或者你的 Gdataset

    # Model Architecture - 设置为当前最佳基线的默认值
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_agg_units', type=int, default=1024)
    parser.add_argument('--gcn_out_units', type=int, default=128) # 基线值
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--layers', type=int, default=3)          # 基线值
    parser.add_argument('--share_param', action='store_true', default=True)
    parser.add_argument('--no-share_param', action='store_false', dest='share_param')
    parser.add_argument('--gcn_agg_norm_symm', action='store_true', default=True)
    parser.add_argument('--no-gcn_agg_norm_symm', action='store_false', dest='gcn_agg_norm_symm')
    parser.add_argument('--nhid1', type=int, default=768)         # 基线值
    parser.add_argument('--nhid2', type=int, default=128)         # 基线值
    parser.add_argument('--use_gate_attention', action='store_true', default=False)

    # Regularization & Dropout - 设置为当前最佳基线的默认值
    parser.add_argument('--dropout', type=float, default=0.3)           # 基线值
    parser.add_argument('--attention_dropout', type=float, default=0.1) # 基线值
    parser.add_argument('--l2_reg_weight', type=float, default=0.0)     # 基线值 (无L2)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # Training Parameters (保持与之前一致或根据需要调整)
    parser.add_argument('--train_lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--train_max_iter', type=int, default=18000)
    parser.add_argument('--train_valid_interval', type=int, default=250)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_patience', type=int, default=10) # 或你之前使用的值
    parser.add_argument('--beta', type=float, default=0.001)

    # Data & Preprocessing
    parser.add_argument('--embedding_mode', type=str, default='pretrained', choices=['pretrained', 'random'])
    parser.add_argument('--num_neighbor', type=int, default=4)

    # Augmentation Parameters - 设置为当前最佳基线的默认值
    parser.add_argument('--use_augmentation', action='store_true', default=True) # 基线值 (开启增强)
    parser.add_argument('--no-use_augmentation', action='store_false', dest='use_augmentation')
    parser.add_argument('--aug_methods', type=str, nargs='+', default=['edge_dropout', 'feature_noise'])
    parser.add_argument('--edge_dropout_rate', type=float, default=0.1)
    parser.add_argument('--add_edge_rate', type=float, default=0.03)
    parser.add_argument('--feature_noise_scale', type=float, default=0.05)
    parser.add_argument('--graph_noise_scale', type=float, default=0.03)
    parser.add_argument('--feature_mask_rate', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)

    # Output & Saving
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--no-save_model', action='store_false', dest='save_model')
    parser.add_argument('--generate_top_predictions', action='store_false', default=False)
    parser.add_argument('--allow-generate_top_predictions', action='store_true', dest='generate_top_predictions')
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--prediction_batch_size', type=int, default=5000)

    # 设置默认值 (如果需要)
    parser.set_defaults(use_gate_attention=False)


    # 解析基础参数
    base_args = parser.parse_args()

    # --- 定义 V4 消融实验配置 ---
    ablation_configs = {
        # 1. 当前最佳基线 (用于确认环境和代码无误)
        # "baseline_v4":          {
        #     # 使用所有默认值:
        #     # gcn_agg_units=1024, gcn_out_units=128, layers=3,
        #     # nhid1=768, nhid2=128, dropout=0.3, etc.
        # },

        # # --- 新增：增加模型尺寸的实验配置 ---

        # 方案 A: 增加 GCMC 输出维度和 FGCN 隐藏维度
        "larger_gcn_out_fgcn_hid": {
            "gcn_out_units": 256,  # 原为 128
            "nhid1": 1024,       # 原为 768
            "nhid2": 256         # 原为 128
            # 其他参数使用默认值
        },


        # 3. 重复测试无数据增强 (来自原始 ablation)
        "no_augmentation_rerun":{
            "use_augmentation": False
        },


        # 方案 B: 显著增加 GCMC 聚合/消息维度
        "larger_gcn_agg_units": {
             "gcn_agg_units": 2048 # 原为 1024
             # 其他参数使用默认值
        },

        # # 方案 C: 增加 GNN 层数 (你的原始 ablation 中已有)
        # "more_layers_rerun":          { # 可以重新运行或检查之前的 more_layers 结果
        #     "layers": 4 # 原为 3
        #     # 其他参数使用默认值
        # },

        # # 方案 D: 组合增大 (显存和计算需求最高)
        # "largest_model_test": {
        #     "gcn_agg_units": 1536, # 适度增加
        #     "gcn_out_units": 256,
        #     "layers": 4,
        #     "nhid1": 1024,
        #     "nhid2": 256
        #     # 其他参数使用默认值
        # },

        # --- 结束新增 ---


        # # 2. 测试更多层数 (来自原始 ablation)
        # # 你可以保留或移除这个，如果上面的 "more_layers_rerun" 足够
        # "more_layers":          {
        #     "layers": 4
        # },


    }

    overall_results = {}
    overall_start_time = time.time()
    # --- 运行 V4 消融实验 ---
    # !!! 修改研究名称 !!!
    study_name = "Ablation Study V4"
    print(f"===== Starting {study_name} (Seed 0 for all configs) =====")
    for name, overrides in ablation_configs.items():
        print(f"\n{'='*25} Starting Ablation Experiment: {name} {'='*25}")
        current_args = copy.deepcopy(base_args) # 每个实验都从基础参数开始

        # 应用当前实验的特定覆盖参数
        for key, value in overrides.items():
            if hasattr(current_args, key):
                setattr(current_args, key, value)
            else:
                print(f"Warning: Ignoring unknown argument override '{key}' for experiment '{name}'")

        # 设置当前实验的保存目录
        current_args.save_dir = os.path.join(current_args.base_save_dir, name)

        try:
            # 运行包含10折交叉验证的实验函数
            avg_auroc, avg_aupr, std_auroc, std_aupr = run_experiment(current_args) # 假设 run_experiment 函数存在且有效
            # 存储聚合结果
            overall_results[name] = {'AUROC': avg_auroc, 'AUPR': avg_aupr, 'AUROC_std': std_auroc, 'AUPR_std': std_aupr }
        except Exception as e:
            # 捕获实验过程中的错误
            print(f"!!!!!! Experiment '{name}' failed catastrophically! !!!!!! Error: {e}")
            traceback.print_exc()
            overall_results[name] = {'AUROC': 'Failed', 'AUPR': 'Failed', 'AUROC_std': 'N/A', 'AUPR_std': 'N/A'}
        print(f"{'='*25} Finished Ablation Experiment: {name} {'='*25}")

    # --- 打印和保存最终总结 ---
    overall_end_time = time.time()
    print(f"\n{'='*30} {study_name} Summary (Seed 0 Results) {'='*30}")

    # 将结果转换为DataFrame
    results_df = pd.DataFrame.from_dict(overall_results, orient='index')

    # 格式化浮点数列以便阅读
    float_cols = ['AUROC', 'AUPR', 'AUROC_std', 'AUPR_std']
    for col in float_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').map(lambda x: f'{x:.4f}' if pd.notna(x) else x)

    # 尝试按AUPR排序（最佳优先）
    try:
        results_df['AUPR_float'] = pd.to_numeric(results_df['AUPR'], errors='coerce')
        results_df = results_df.sort_values('AUPR_float', ascending=False, na_position='last').drop(columns=['AUPR_float'])
    except Exception as e:
        print(f"Note: Could not sort results by AUPR. Error: {e}")

    # 打印最终总结表
    print(results_df.to_string())

    # 确保保存总结的基础目录存在
    if not os.path.isdir(base_args.base_save_dir):
        os.makedirs(base_args.base_save_dir, exist_ok=True)

    # 定义总结CSV文件的路径
    # !!! 修改总结文件名 !!!
    summary_path = os.path.join(base_args.base_save_dir, "ablation_summary_v4_seed0.csv")

    # 保存总结DataFrame到CSV文件
    try:
        results_df.to_csv(summary_path)
        print(f"\nAblation summary saved to {summary_path}")
    except Exception as e:
        print(f"!!! Error saving ablation summary: {e}")

    # 打印消融研究的总时长
    print(f"Total ablation study duration: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"{study_name} finished.")