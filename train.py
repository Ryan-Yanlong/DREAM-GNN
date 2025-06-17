import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from model import Net
from evaluation import evaluate
from data_loader import DrugDataLoader
from utils import MetricLogger, common_loss, setup_seed
from augmentation import augment_graph_data
import torch.nn.functional as F

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Smooth target: 1 -> 1-smoothing, 0 -> smoothing
        smooth_target = target * (1 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy_with_logits(pred, smooth_target)


def get_top_novel_predictions(args, model, dataset, cv_idx, top_k=200):
    """
    Get top K predictions (drug-disease pairs) that are not present in real data
    
    Args:
        args: Arguments containing device settings
        model: Trained model
        dataset: DrugDataLoader instance
        cv_idx: Cross-validation fold index
        top_k: Number of top predictions to return (default: 200)
        
    Returns:
        top_pairs: DataFrame containing top predicted drug-disease pairs
    """
    print(f"Generating top {top_k} novel predictions for fold {cv_idx+1}...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get real association matrix
    ground_truth = dataset.association_matrix
    
    # Get graph data for this fold
    graph_data = dataset.get_graph_data_for_training(cv_idx)
    
    # Create all possible drug-disease pairs that don't exist in real data
    novel_pairs = []
    for drug_id in range(dataset.num_drug):
        for disease_id in range(dataset.num_disease):
            if ground_truth[drug_id, disease_id] == 0:  # Not in real data
                novel_pairs.append((drug_id, disease_id))
    
    print(f"Found {len(novel_pairs)} potential novel drug-disease pairs.")
    
    # Create batches for prediction to avoid memory issues
    batch_size = 5000  # Reduce batch size to lower memory usage
    num_batches = len(novel_pairs) // batch_size + (1 if len(novel_pairs) % batch_size > 0 else 0)
    
    all_predictions = []
    
    with th.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(novel_pairs))
            batch_pairs = novel_pairs[start_idx:end_idx]
            
            try:
                # Create rating pairs in format required by dataset._generate_dec_graph
                rating_pairs = (
                    np.array([p[0] for p in batch_pairs], dtype=np.int64),
                    np.array([p[1] for p in batch_pairs], dtype=np.int64)
                )
                
                # Generate decoder graph for these pairs
                dec_graph = dataset._generate_dec_graph(rating_pairs)
                dec_graph = dec_graph.to(args.device)
                
                # Use training encoder graph containing learned node representations
                enc_graph = graph_data['train_enc_graph']
                
                # Get all other required inputs
                drug_graph = graph_data['drug_graph']
                drug_sim_feat = graph_data['drug_sim_features']
                drug_feat = graph_data['drug_features']
                dis_graph = graph_data['disease_graph']
                dis_sim_feat = graph_data['disease_sim_features']
                dis_feat = graph_data['disease_features']
                drug_feature_graph = graph_data['drug_feature_graph']
                disease_feature_graph = graph_data['disease_feature_graph']
                
                # Forward pass
                pred_ratings, _, _, _, _ = model(
                    enc_graph, dec_graph,
                    drug_graph, drug_sim_feat, drug_feat,
                    dis_graph, dis_sim_feat, dis_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
                # Extract predictions - ensure sigmoid is applied to get probabilities
                pred_scores = th.sigmoid(pred_ratings.squeeze(-1)).cpu().numpy()
                
                # Store predictions
                for j, (drug_id, disease_id) in enumerate(batch_pairs):
                    if j < len(pred_scores):  # Ensure index is valid
                        all_predictions.append({
                            'drug_id': drug_id,
                            'disease_id': disease_id,
                            'score': float(pred_scores[j])
                        })
                
                print(f"Processed batch {i+1}/{num_batches}")
                
            except Exception as e:
                print(f"Error processing batch {i+1}/{num_batches}: {str(e)}")
                # Continue to next batch without terminating entire process
                continue
    
    # Convert to DataFrame and sort by score
    if not all_predictions:
        print("Warning: No predictions were generated during batch processing.")
        # Return empty DataFrame
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])
    
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Map IDs to drug and disease names if available
    if hasattr(dataset, 'drug_ids') and dataset.drug_ids is not None:
        try:
            drug_id_to_name = {i: name for i, name in enumerate(dataset.drug_ids)}
            pred_df['drug_name'] = pred_df['drug_id'].map(drug_id_to_name)
        except Exception as e:
            print(f"Error mapping drug IDs to names: {str(e)}")
    
    # Get top K predictions
    top_pred_df = pred_df.head(top_k)
    
    # Save to CSV
    try:
        csv_path = os.path.join(args.save_dir, f"top{top_k}_novel_predictions_fold{cv_idx+1}.csv")
        top_pred_df.to_csv(csv_path, index=False)
        print(f"Top {top_k} novel predictions saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")
    
    return top_pred_df


def train(args, dataset, cv):
    """
    Model training process using cross-validation fold-specific graph data:
      - Set input dimensions and initialize model
      - Get fold-specific graph structures from dataset
      - Enhance training data using data augmentation techniques
      - Periodically evaluate model performance and save best model

    Args:
      args: Command line arguments containing hyperparameters, device settings, etc.
      dataset: DrugDataLoader instance containing data and features
      cv: Current cross-validation fold number

    Returns:
      best_auroc: Best test AUROC
      best_aupr: Best test AUPR
    """
    # Set input dimensions
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    
    # Use rating_vals from dataset to ensure consistency
    args.rating_vals = dataset.cv_data_dict[cv][2]  # Get possible association values from current fold
    print(f"[Model] Using rating values: {args.rating_vals}")

    # Get fold-specific graph data
    cv_data = dataset.data_cv[cv]
    fold_specific_graphs = dataset.cv_specific_graphs[cv]
    
    # Extract graph structures needed for training
    drug_graph = fold_specific_graphs['drug_graph'].to(args.device)
    dis_graph = fold_specific_graphs['disease_graph'].to(args.device)
    drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(args.device)
    disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(args.device)

    # Get feature data
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_feat = dataset.drug_feature.to(args.device)
    dis_feat = dataset.disease_feature.to(args.device)

    # Get training and testing data
    train_gt_ratings = cv_data['train'][2].to(args.device)
    train_enc_graph = cv_data['train'][0].int().to(args.device)
    train_dec_graph = cv_data['train'][1].int().to(args.device)
    
    # Build training and testing data dictionaries (for evaluation)
    train_data_dict = {'test': cv_data['train']}
    test_data_dict = {'test': cv_data['test']}

    # Build model, loss function and optimizer
    model = Net(args=args).to(args.device)
    
    # Choose loss function (optionally enable label smoothing)
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        rel_loss_fn = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
        print(f"Using Label Smoothing BCE Loss with smoothing={args.label_smoothing}")
    else:
        rel_loss_fn = nn.BCEWithLogitsLoss()
        print("Using standard BCE Loss")
        
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    print("Network loaded and initialized.")

    # Initialize logger
    test_loss_logger = MetricLogger(
        ['iter', 'loss', 'train_auroc', 'train_aupr', 'test_auroc', 'test_aupr'],
        ['%d', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'],
        os.path.join(args.save_dir, f'test_metric{args.save_id}.csv')
    )

    print("Start training...")
    best_aupr = -1.0
    best_auroc = 0.0
    best_iter = 0
    best_train_aupr = 0.0
    best_train_auroc = 0.0

    # Learning rate scheduler - adjust learning rate based on validation performance
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=500, factor=0.5)

    # Data augmentation parameters
    aug_methods = args.aug_methods if hasattr(args, 'aug_methods') else ['edge_dropout', 'feature_noise']
    aug_params = {
        'edge_dropout_rate': args.edge_dropout_rate if hasattr(args, 'edge_dropout_rate') else 0.1,
        'feature_noise_scale': args.feature_noise_scale if hasattr(args, 'feature_noise_scale') else 0.05,
        'graph_noise_scale': args.graph_noise_scale if hasattr(args, 'graph_noise_scale') else 0.02,
        'add_edge_rate': args.add_edge_rate if hasattr(args, 'add_edge_rate') else 0.03,
        'feature_mask_rate': args.feature_mask_rate if hasattr(args, 'feature_mask_rate') else 0.1,
        'mixup_alpha': args.mixup_alpha if hasattr(args, 'mixup_alpha') else 0.2
    }

    start_time = time.perf_counter()
    for iter_idx in range(1, args.train_max_iter):
        model.train()
        Two_Stage = False  # Modify to two-stage training if needed

        # Prepare data to be augmented
        graph_data_to_augment = {
            'enc_graph': train_enc_graph,
            'drug_graph': drug_graph,
            'disease_graph': dis_graph,
            'drug_feature_graph': drug_feature_graph,
            'disease_feature_graph': disease_feature_graph,
            'drug_feat': drug_feat,
            'disease_feat': dis_feat,
            'drug_sim_feat': drug_sim_feat,
            'disease_sim_feat': dis_sim_feat
        }
        
        # Apply data augmentation
        augmented_data = augment_graph_data(graph_data_to_augment, aug_methods, aug_params)
        aug_train_enc_graph = augmented_data['enc_graph']
        aug_train_dec_graph = train_dec_graph  # Decoder graph not augmented
        aug_drug_graph = augmented_data['drug_graph']
        aug_disease_graph = augmented_data['disease_graph']
        aug_drug_feature_graph = augmented_data['drug_feature_graph']
        aug_disease_feature_graph = augmented_data['disease_feature_graph']
        aug_drug_feat = augmented_data['drug_feat']
        aug_dis_feat = augmented_data['disease_feat']
        aug_drug_sim_feat = augmented_data['drug_sim_feat']
        aug_dis_sim_feat = augmented_data['disease_sim_feat']

        # Forward pass (using augmented data)
        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out = model(
            aug_train_enc_graph, aug_train_dec_graph,
            aug_drug_graph, aug_drug_sim_feat, aug_drug_feat,
            aug_disease_graph, aug_dis_sim_feat, aug_dis_feat,
            aug_drug_feature_graph, aug_disease_feature_graph, Two_Stage
        )
        pred_ratings = pred_ratings.squeeze(-1)

        # Calculate loss
        loss_com_drug = common_loss(drug_out, drug_sim_out)
        loss_com_dis = common_loss(dis_out, dis_sim_out)
        rel_loss_val = rel_loss_fn(pred_ratings, train_gt_ratings)
        
        # Total loss = association prediction loss + common representation learning loss
        total_loss = rel_loss_val + args.beta * (loss_com_drug + loss_com_dis)

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        # Periodically evaluate model performance
        if iter_idx % args.train_valid_interval == 0:
            model.eval()
            with th.no_grad():
                # Evaluate training set performance
                train_auroc, train_aupr = evaluate(
                    args, model, train_data_dict,
                    drug_graph, drug_feat, drug_sim_feat,
                    dis_graph, dis_feat, dis_sim_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
                # Evaluate test set performance
                test_auroc, test_aupr = evaluate(
                    args, model, test_data_dict,
                    drug_graph, drug_feat, drug_sim_feat,
                    dis_graph, dis_feat, dis_sim_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
            # Update learning rate
            scheduler.step(test_aupr)
            
            # Log metrics
            test_loss_logger.log(
                iter=iter_idx, 
                loss=total_loss.item(), 
                train_auroc=train_auroc, 
                train_aupr=train_aupr,
                test_auroc=test_auroc, 
                test_aupr=test_aupr
            )
            
            # Print progress
            log_str = (f"Iter={iter_idx:5d}, Loss={total_loss.item():.4f}, "
                       f"Train: AUROC={train_auroc:.4f}, AUPR={train_aupr:.4f}, "
                       f"Test: AUROC={test_auroc:.4f}, AUPR={test_aupr:.4f}")
            print(log_str)
            
            # Save best model
            if test_aupr > best_aupr:
                best_aupr = test_aupr
                best_auroc = test_auroc
                best_train_aupr = train_aupr
                best_train_auroc = train_auroc
                best_iter = iter_idx
                
                if args.save_model:
                    model_path = os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")
                    th.save(model.state_dict(), model_path)
                    
    # Training finished, calculate total time
    elapsed_time = time.perf_counter() - start_time
    print("Running time:", time.strftime("%H:%M:%S", time.gmtime(round(elapsed_time))))
    test_loss_logger.close()

    # Output best results
    print(f"Best iteration: {best_iter} with metrics:")
    print(f"  Train set - AUROC: {best_train_auroc:.4f}, AUPR: {best_train_aupr:.4f}")
    print(f"  Test set  - AUROC: {best_auroc:.4f}, AUPR: {best_aupr:.4f}")

    # Save best metrics
    best_metrics_path = os.path.join(args.save_dir, f"best_metric{args.save_id}.csv")
    with open(best_metrics_path, 'w') as f:
        f.write("iter,train_auroc,train_aupr,test_auroc,test_aupr\n")
        f.write(f"{best_iter},{best_train_auroc:.4f},{best_train_aupr:.4f},{best_auroc:.4f},{best_aupr:.4f}\n")

    # After training, generate top 200 novel predictions using best model
    if args.save_model and args.generate_top_predictions:
        print("\nGenerating novel predictions using best model...")
        # Load best model
        best_model = Net(args=args).to(args.device)
        best_model.load_state_dict(th.load(os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")))
        # Get top K novel predictions
        top_predictions = get_top_novel_predictions(args, best_model, dataset, cv, top_k=args.top_k)
        print(f"Top 5 novel predictions:\n{top_predictions.head(5)}")
        
        # Create more detailed output with additional information
        detailed_output_path = os.path.join(args.save_dir, f"detailed_top_predictions_fold{args.save_id}.csv")
        
        # Include drug names in detailed output if available
        if hasattr(dataset, 'drug_ids') and dataset.drug_ids is not None:
            for i, row in top_predictions.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                drug_name = row['drug_name'] if 'drug_name' in row else f"Drug_{drug_id}"
                print(f"Rank {i+1}: Drug ID {drug_id} ({drug_name}) - Disease ID {disease_id}, Score: {row['score']:.4f}")
        else:
            for i, row in top_predictions.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                print(f"Rank {i+1}: Drug ID {drug_id} - Disease ID {disease_id}, Score: {row['score']:.4f}")
                
    return best_auroc, best_aupr


###############################################################################
# Main entry point: parse arguments, use specified random seed list, 
# load data, perform cross-validation training, etc.
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaDR with Data Augmentation and L2 Regularization')
    parser.add_argument('--device', default='0', type=int,
                        help='Run device, e.g. "--device 0", set to --device -1 for CPU')
    parser.add_argument('--save_dir', type=str, help='Log save directory')
    parser.add_argument('--save_id', type=int, help='Log save ID')
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=1024)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=128)
    parser.add_argument('--train_max_iter', type=int, default=18000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=250)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--nhid1', type=int, default=768)
    parser.add_argument('--nhid2', type=int, default=128)
    parser.add_argument('--train_lr', type=float, default=0.002)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--share_param', default=True, action='store_true')
    parser.add_argument('--data_name', default='Gdataset', type=str)
    parser.add_argument('--num_neighbor', type=int, default=4, help='default number of neighbors')
    parser.add_argument('--beta', type=float, default=0.001, help='weight for common representation learning loss')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay coefficient')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0000, help='L2 regularization weight')
    parser.add_argument('--attention_dropout', type=float, default=0.1, 
                        help='dropout rate in attention mechanism')
    parser.add_argument('--embedding_mode', type=str, default='pretrained', choices=['pretrained', 'random'],
                        help='choose pretrained embeddings or random initialization')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--aug_methods', type=str, nargs='+', 
                        default=['edge_dropout', 'feature_noise'],
                        choices=['edge_dropout', 'add_random_edges', 'feature_noise', 
                                 'graph_noise', 'feature_masking', 'mix_up'],
                        help='data augmentation methods to use')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.1, help='edge dropout probability')
    parser.add_argument('--add_edge_rate', type=float, default=0.03, help='ratio of randomly added edges')
    parser.add_argument('--feature_noise_scale', type=float, default=0.05, help='feature noise standard deviation')
    parser.add_argument('--graph_noise_scale', type=float, default=0.03, help='graph structure noise standard deviation')
    parser.add_argument('--feature_mask_rate', type=float, default=0.1, help='feature masking rate')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup parameter')
    parser.add_argument('--save_model', action='store_true', help='whether to save best model')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing degree, 0 means not used')
    parser.add_argument('--generate_top_predictions', action='store_true', default=False, 
                        help='generate top novel predictions after training')
    parser.add_argument('--top_k', type=int, default=200, 
                        help='number of top predictions to generate')
    
    parser.set_defaults(use_gate_attention=False)

    args = parser.parse_args()
    print(args)

    # Use fixed random seed list
    fixed_seeds = [77, 31415, 888, 1001, 9999, 0, 42, 123, 2024, 7]
   
    # Set device
    if args.device >= 0:
        args.device = f"cuda:{args.device}" if th.cuda.is_available() else "cpu"
    else:
        args.device = "cpu"
    print(f"Using device: {args.device}")

    # Record results for all experiments
    all_results = []
    all_auroc = []
    all_aupr = []

    # Run 10 experiments with specified random seed list
    for exp_idx, seed in enumerate(fixed_seeds):
        print(f"======== Experiment {exp_idx+1}/10 with seed {seed} ========")
        
        # Set random seed for current experiment
        setup_seed(seed)
        
        # Create experiment directory
        exp_dir = os.path.join("seed_experiments", f"seed_{seed}")
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        args.save_dir = exp_dir

        # Initialize data loader
        dataset = DrugDataLoader(args.data_name, args.device,
                                symm=args.gcn_agg_norm_symm,
                                k=args.num_neighbor,
                                use_augmentation=args.use_augmentation,
                                aug_params={
                                    'edge_dropout_rate': args.edge_dropout_rate,    
                                    'feature_noise_scale': args.feature_noise_scale,
                                    'graph_noise_scale': args.graph_noise_scale,
                                    'add_edge_rate': args.add_edge_rate,
                                    'feature_mask_rate': args.feature_mask_rate,
                                })
        dataset.embedding_mode = args.embedding_mode
        print("Loading dataset finished ...\n")

        # Perform 10-fold cross-validation
        fold_results = []
        for cv in range(10):
            args.save_id = cv + 1
            print("============== Fold {} ==============".format(cv + 1))
            # Train and evaluate model
            auroc, aupr = train(args, dataset, cv)
            fold_results.append((auroc, aupr))
            
        # Calculate average performance for current experiment
        avg_auroc = sum(x[0] for x in fold_results) / len(fold_results)
        avg_aupr = sum(x[1] for x in fold_results) / len(fold_results)
        
        # Save experiment results
        all_results.append({
            'seed': seed,
            'avg_auroc': avg_auroc,
            'avg_aupr': avg_aupr,
            'fold_results': fold_results
        })
        all_auroc.append(avg_auroc)
        all_aupr.append(avg_aupr)
        
        # Record current experiment results to file
        results_path = os.path.join(exp_dir, "experiment_results.csv")
        with open(results_path, 'w') as f:
            f.write("fold,auroc,aupr\n")
            for i, (fold_auroc, fold_aupr) in enumerate(fold_results):
                f.write(f"{i+1},{fold_auroc:.4f},{fold_aupr:.4f}\n")
            f.write(f"average,{avg_auroc:.4f},{avg_aupr:.4f}\n")
        
        print(f"Experiment {exp_idx+1} (Seed {seed}) - Avg AUROC: {avg_auroc:.4f}, Avg AUPR: {avg_aupr:.4f}")
    
    # Calculate overall statistics for all experiments
    overall_avg_auroc = sum(all_auroc) / len(all_auroc)
    overall_avg_aupr = sum(all_aupr) / len(all_aupr)
    
    # Calculate standard deviation
    auroc_std = np.std(all_auroc)
    aupr_std = np.std(all_aupr)
    
    # Find best and worst experiments
    best_exp_idx = np.argmax(all_auroc)
    worst_exp_idx = np.argmin(all_auroc)
    
    # Output overall statistics
    print("\n===== OVERALL RESULTS =====")
    print(f"Overall Average - AUROC: {overall_avg_auroc:.4f} ± {auroc_std:.4f}, AUPR: {overall_avg_aupr:.4f} ± {aupr_std:.4f}")
    print(f"Best Result (Seed {fixed_seeds[best_exp_idx]}) - AUROC: {all_auroc[best_exp_idx]:.4f}, AUPR: {all_aupr[best_exp_idx]:.4f}")
    print(f"Worst Result (Seed {fixed_seeds[worst_exp_idx]}) - AUROC: {all_auroc[worst_exp_idx]:.4f}, AUPR: {all_aupr[worst_exp_idx]:.4f}")
    
    # Save overall experiment summary
    summary_path = os.path.join("seed_experiments", "summary_results.csv")
    with open(summary_path, 'w') as f:
        f.write("experiment,seed,avg_auroc,avg_aupr\n")
        for i, res in enumerate(all_results):
            f.write(f"{i+1},{res['seed']},{res['avg_auroc']:.4f},{res['avg_aupr']:.4f}\n")
        f.write(f"overall,NA,{overall_avg_auroc:.4f},{overall_avg_aupr:.4f}\n")
        f.write(f"std,NA,{auroc_std:.4f},{aupr_std:.4f}\n")