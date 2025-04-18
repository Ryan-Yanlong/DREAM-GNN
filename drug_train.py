import os
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from model import Net
from evaluate import evaluate
from data import DrugDataLoader
from utils import MetricLogger, common_loss, setup_seed
from graph_augmentation import GraphAugmentation, augment_graph_data
import torch.nn.functional as F

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # 平滑目标：1 -> 1-smoothing, 0 -> smoothing
        smooth_target = target * (1 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy_with_logits(pred, smooth_target)


def get_top_novel_predictions(args, model, dataset, cv_idx, top_k=200):
    """
    获取未出现在真实数据中的前K个预测（药物-疾病对）
    
    参数：
        args: 包含设备设置的参数
        model: 训练好的模型
        dataset: DrugDataLoader实例
        cv_idx: 交叉验证折索引
        top_k: 要返回的顶部预测数量（默认：200）
        
    返回：
        top_pairs: 包含顶部预测药物-疾病对的DataFrame
    """
    print(f"生成第{cv_idx+1}折的前{top_k}个新型预测...")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 获取真实关联矩阵
    ground_truth = dataset.association_matrix
    
    # 获取此折的图数据
    graph_data = dataset.get_graph_data_for_training(cv_idx)
    
    # 创建所有在真实数据中不存在的可能药物-疾病对
    novel_pairs = []
    for drug_id in range(dataset.num_drug):
        for disease_id in range(dataset.num_disease):
            if ground_truth[drug_id, disease_id] == 0:  # 不在真实数据中
                novel_pairs.append((drug_id, disease_id))
    
    print(f"找到{len(novel_pairs)}个潜在的新型药物-疾病对。")
    
    # 创建批次进行预测，避免内存问题
    batch_size = 5000  # 减小批次大小以降低内存使用
    num_batches = len(novel_pairs) // batch_size + (1 if len(novel_pairs) % batch_size > 0 else 0)
    
    all_predictions = []
    
    with th.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(novel_pairs))
            batch_pairs = novel_pairs[start_idx:end_idx]
            
            try:
                # 创建格式符合dataset._generate_dec_graph要求的评分对
                rating_pairs = (
                    np.array([p[0] for p in batch_pairs], dtype=np.int64),
                    np.array([p[1] for p in batch_pairs], dtype=np.int64)
                )
                
                # 为这些对生成解码器图
                dec_graph = dataset._generate_dec_graph(rating_pairs)
                dec_graph = dec_graph.to(args.device)
                
                # 使用包含学习到的节点表示的训练编码器图
                enc_graph = graph_data['train_enc_graph']
                
                # 获取所有其他必需的输入
                drug_graph = graph_data['drug_graph']
                drug_sim_feat = graph_data['drug_sim_features']
                drug_feat = graph_data['drug_features']
                dis_graph = graph_data['disease_graph']
                dis_sim_feat = graph_data['disease_sim_features']
                dis_feat = graph_data['disease_features']
                drug_feature_graph = graph_data['drug_feature_graph']
                disease_feature_graph = graph_data['disease_feature_graph']
                
                # 前向传播
                pred_ratings, _, _, _, _ = model(
                    enc_graph, dec_graph,
                    drug_graph, drug_sim_feat, drug_feat,
                    dis_graph, dis_sim_feat, dis_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
                # 提取预测 - 确保应用sigmoid获取概率
                pred_scores = th.sigmoid(pred_ratings.squeeze(-1)).cpu().numpy()
                
                # 存储预测
                for j, (drug_id, disease_id) in enumerate(batch_pairs):
                    if j < len(pred_scores):  # 确保索引有效
                        all_predictions.append({
                            'drug_id': drug_id,
                            'disease_id': disease_id,
                            'score': float(pred_scores[j])
                        })
                
                print(f"处理批次 {i+1}/{num_batches}")
                
            except Exception as e:
                print(f"处理批次 {i+1}/{num_batches} 时出错: {str(e)}")
                # 继续下一个批次而不终止整个过程
                continue
    
    # 转换为DataFrame并按分数排序
    if not all_predictions:
        print("警告: 没有生成任何预测。请检查批处理过程中的错误。")
        # 返回空DataFrame
        return pd.DataFrame(columns=['drug_id', 'disease_id', 'score'])
    
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # 如果可用，将ID映射到药物和疾病名称
    if hasattr(dataset, 'drug_ids') and dataset.drug_ids is not None:
        try:
            drug_id_to_name = {i: name for i, name in enumerate(dataset.drug_ids)}
            pred_df['drug_name'] = pred_df['drug_id'].map(drug_id_to_name)
        except Exception as e:
            print(f"映射药物ID到名称时出错: {str(e)}")
    
    # 获取前K个预测
    top_pred_df = pred_df.head(top_k)
    
    # 保存到CSV
    try:
        csv_path = os.path.join(args.save_dir, f"top{top_k}_novel_predictions_fold{cv_idx+1}.csv")
        top_pred_df.to_csv(csv_path, index=False)
        print(f"前{top_k}个新型预测已保存到 {csv_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")
    
    return top_pred_df
def train(args, dataset, cv):
    """
    模型训练流程，使用交叉验证折特定的图数据：
      - 设置输入维度并初始化模型；
      - 从数据集获取特定于当前交叉验证折的图结构；
      - 使用数据增强技术增强训练数据；
      - 定期评估模型性能并保存最佳模型。

    参数：
      args: 命令行参数，包含超参数、设备设置等。
      dataset: DrugDataLoader 实例，包含数据和特征。
      cv: 当前交叉验证折编号。

    返回：
      best_auroc: 最佳测试 AUROC。
      best_aupr: 最佳测试 AUPR。
    """
    # 设置输入维度
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    
    # 使用数据集中的rating_vals，确保一致性
    args.rating_vals = dataset.cv_data_dict[cv][2]  # 从当前折获取可能的关联值
    print(f"[Model] Using rating values: {args.rating_vals}")

    # 获取该交叉验证折的特定图数据
    cv_data = dataset.data_cv[cv]
    fold_specific_graphs = dataset.cv_specific_graphs[cv]
    
    # 提取训练所需的图结构
    drug_graph = fold_specific_graphs['drug_graph'].to(args.device)
    dis_graph = fold_specific_graphs['disease_graph'].to(args.device)
    drug_feature_graph = fold_specific_graphs['drug_feature_graph'].to(args.device)
    disease_feature_graph = fold_specific_graphs['disease_feature_graph'].to(args.device)

    # 获取特征数据
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_feat = dataset.drug_feature.to(args.device)
    dis_feat = dataset.disease_feature.to(args.device)

    # 获取训练和测试数据
    train_gt_ratings = cv_data['train'][2].to(args.device)
    train_enc_graph = cv_data['train'][0].int().to(args.device)
    train_dec_graph = cv_data['train'][1].int().to(args.device)
    
    # 构建训练和测试数据字典（用于评估）
    train_data_dict = {'test': cv_data['train']}
    test_data_dict = {'test': cv_data['test']}

    # 构建模型、损失函数和优化器
    model = Net(args=args).to(args.device)
    
    # 选择损失函数（可选启用标签平滑）
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        rel_loss_fn = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
        print(f"Using Label Smoothing BCE Loss with smoothing={args.label_smoothing}")
    else:
        rel_loss_fn = nn.BCEWithLogitsLoss()
        print("Using standard BCE Loss")
        
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    print("Network loaded and initialized.")

    # 初始化日志记录器
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

    # 学习率调度器 - 根据验证集性能调整学习率
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=500, factor=0.5)

    # 数据增强参数
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
        Two_Stage = False  # 根据需要修改为两阶段训练

        # 准备待增强数据
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
        
        # 应用数据增强
        augmented_data = augment_graph_data(graph_data_to_augment, aug_methods, aug_params)
        aug_train_enc_graph = augmented_data['enc_graph']
        aug_train_dec_graph = train_dec_graph  # 解码器图不增强
        aug_drug_graph = augmented_data['drug_graph']
        aug_disease_graph = augmented_data['disease_graph']
        aug_drug_feature_graph = augmented_data['drug_feature_graph']
        aug_disease_feature_graph = augmented_data['disease_feature_graph']
        aug_drug_feat = augmented_data['drug_feat']
        aug_dis_feat = augmented_data['disease_feat']
        aug_drug_sim_feat = augmented_data['drug_sim_feat']
        aug_dis_sim_feat = augmented_data['disease_sim_feat']

        # 前向传播（使用增强后的数据）
        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out = model(
            aug_train_enc_graph, aug_train_dec_graph,
            aug_drug_graph, aug_drug_sim_feat, aug_drug_feat,
            aug_disease_graph, aug_dis_sim_feat, aug_dis_feat,
            aug_drug_feature_graph, aug_disease_feature_graph, Two_Stage
        )
        pred_ratings = pred_ratings.squeeze(-1)

        # 计算损失
        loss_com_drug = common_loss(drug_out, drug_sim_out)
        loss_com_dis = common_loss(dis_out, dis_sim_out)
        rel_loss_val = rel_loss_fn(pred_ratings, train_gt_ratings)
        
        # 总损失 = 关联预测损失 + 公共表示学习损失
        total_loss = rel_loss_val + args.beta * (loss_com_drug + loss_com_dis)

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        # 定期评估模型性能
        if iter_idx % args.train_valid_interval == 0:
            model.eval()
            with th.no_grad():
                # 评估训练集性能
                train_auroc, train_aupr = evaluate(
                    args, model, train_data_dict,
                    drug_graph, drug_feat, drug_sim_feat,
                    dis_graph, dis_feat, dis_sim_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
                # 评估测试集性能
                test_auroc, test_aupr = evaluate(
                    args, model, test_data_dict,
                    drug_graph, drug_feat, drug_sim_feat,
                    dis_graph, dis_feat, dis_sim_feat,
                    drug_feature_graph, disease_feature_graph
                )
                
            # 更新学习率
            scheduler.step(test_aupr)
            
            # 记录日志
            test_loss_logger.log(
                iter=iter_idx, 
                loss=total_loss.item(), 
                train_auroc=train_auroc, 
                train_aupr=train_aupr,
                test_auroc=test_auroc, 
                test_aupr=test_aupr
            )
            
            # 打印进度
            log_str = (f"Iter={iter_idx:5d}, Loss={total_loss.item():.4f}, "
                       f"Train: AUROC={train_auroc:.4f}, AUPR={train_aupr:.4f}, "
                       f"Test: AUROC={test_auroc:.4f}, AUPR={test_aupr:.4f}")
            print(log_str)
            
            # 保存最佳模型
            if test_aupr > best_aupr:
                best_aupr = test_aupr
                best_auroc = test_auroc
                best_train_aupr = train_aupr
                best_train_auroc = train_auroc
                best_iter = iter_idx
                
                if args.save_model:
                    model_path = os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")
                    th.save(model.state_dict(), model_path)
                    
    # 训练结束，计算总用时
    elapsed_time = time.perf_counter() - start_time
    print("Running time:", time.strftime("%H:%M:%S", time.gmtime(round(elapsed_time))))
    test_loss_logger.close()

    # 输出最佳结果
    print(f"Best iteration: {best_iter} with metrics:")
    print(f"  Train set - AUROC: {best_train_auroc:.4f}, AUPR: {best_train_aupr:.4f}")
    print(f"  Test set  - AUROC: {best_auroc:.4f}, AUPR: {best_aupr:.4f}")

    # 保存最佳指标
    best_metrics_path = os.path.join(args.save_dir, f"best_metric{args.save_id}.csv")
    with open(best_metrics_path, 'w') as f:
        f.write("iter,train_auroc,train_aupr,test_auroc,test_aupr\n")
        f.write(f"{best_iter},{best_train_auroc:.4f},{best_train_aupr:.4f},{best_auroc:.4f},{best_aupr:.4f}\n")

    # 训练结束后，使用最佳模型生成前200个新型预测
    if args.save_model and args.generate_top_predictions:
        print("\n使用最佳模型生成新型预测...")
        # 加载最佳模型
        best_model = Net(args=args).to(args.device)
        best_model.load_state_dict(th.load(os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")))
        # 获取前K个新型预测
        top_predictions = get_top_novel_predictions(args, best_model, dataset, cv, top_k=args.top_k)
        print(f"前5个新型预测:\n{top_predictions.head(5)}")
        
        # 创建更详细的输出，包含额外信息
        detailed_output_path = os.path.join(args.save_dir, f"detailed_top_predictions_fold{args.save_id}.csv")
        
        # 如果药物名称可用，在详细输出中包含它们
        if hasattr(dataset, 'drug_ids') and dataset.drug_ids is not None:
            for i, row in top_predictions.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                drug_name = row['drug_name'] if 'drug_name' in row else f"Drug_{drug_id}"
                print(f"排名 {i+1}: 药物ID {drug_id} ({drug_name}) - 疾病ID {disease_id}, 得分: {row['score']:.4f}")
        else:
            for i, row in top_predictions.iterrows():
                drug_id = row['drug_id']
                disease_id = row['disease_id']
                print(f"排名 {i+1}: 药物ID {drug_id} - 疾病ID {disease_id}, 得分: {row['score']:.4f}")
                
    return best_auroc, best_aupr

###############################################################################
# 主函数入口：解析参数、使用指定的随机种子列表、加载数据、进行交叉验证训练等
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaDR with Data Augmentation and L2 Regularization')
    parser.add_argument('--device', default='0', type=int,
                        help='运行设备，例如 "--device 0"，若使用 CPU 则设置为 --device -1')
    parser.add_argument('--save_dir', type=str, help='日志保存目录')
    parser.add_argument('--save_id', type=int, help='日志保存ID')
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=1024)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=128)
    parser.add_argument('--train_max_iter', type=int, default=9000)
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
    parser.add_argument('--beta', type=float, default=0.001, help='公共表示学习损失权重')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='优化器权重衰减系数')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0000, help='L2正则化权重')
    parser.add_argument('--attention_dropout', type=float, default=0.1, 
                        help='注意力机制中的dropout率')
    parser.add_argument('--embedding_mode', type=str, default='pretrained', choices=['pretrained', 'random'],
                        help='选择使用预训练embedding还是随机初始化')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='是否使用数据增强')
    parser.add_argument('--aug_methods', type=str, nargs='+', 
                        default=['edge_dropout', 'feature_noise'],
                        choices=['edge_dropout', 'add_random_edges', 'feature_noise', 
                                 'graph_noise', 'feature_masking', 'mix_up'],
                        help='要使用的数据增强方法')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.1, help='边丢弃概率')
    parser.add_argument('--add_edge_rate', type=float, default=0.03, help='随机添加边的比例')
    parser.add_argument('--feature_noise_scale', type=float, default=0.05, help='特征噪声标准差')
    parser.add_argument('--graph_noise_scale', type=float, default=0.03, help='图结构噪声标准差')
    parser.add_argument('--feature_mask_rate', type=float, default=0.1, help='特征遮蔽率')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup参数')
    parser.add_argument('--save_model', action='store_true', help='是否保存最佳模型')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑程度，0表示不使用')
    parser.add_argument('--generate_top_predictions', action='store_true', default=False, 
                        help='训练后生成顶部新型预测')
    parser.add_argument('--top_k', type=int, default=200, 
                        help='要生成的顶部预测数量')
    
    parser.set_defaults(use_gate_attention=False)

    args = parser.parse_args()
    print(args)

    # 使用固定的随机种子列表
    fixed_seeds = [77, 31415, 888, 1001, 9999, 0, 42, 123, 2024, 7]
   
    # 设置设备
    if args.device >= 0:
        args.device = f"cuda:{args.device}" if th.cuda.is_available() else "cpu"
    else:
        args.device = "cpu"
    print(f"Using device: {args.device}")

    # 记录所有实验的结果
    all_results = []
    all_auroc = []
    all_aupr = []

    # 使用指定的随机种子列表进行10次实验
    for exp_idx, seed in enumerate(fixed_seeds):
        print(f"======== Experiment {exp_idx+1}/10 with seed {seed} ========")
        
        # 设置当前实验的随机种子
        setup_seed(seed)
        
        # 创建实验目录
        exp_dir = os.path.join("seed_experiments", f"seed_{seed}")
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        args.save_dir = exp_dir

        # 初始化数据加载器
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

        # 进行10折交叉验证
        fold_results = []
        for cv in range(10):
            args.save_id = cv + 1
            print("============== Fold {} ==============".format(cv + 1))
            # 训练和评估模型
            auroc, aupr = train(args, dataset, cv)
            fold_results.append((auroc, aupr))
            
        # 计算当前实验的平均性能
        avg_auroc = sum(x[0] for x in fold_results) / len(fold_results)
        avg_aupr = sum(x[1] for x in fold_results) / len(fold_results)
        
        # 保存实验结果
        all_results.append({
            'seed': seed,
            'avg_auroc': avg_auroc,
            'avg_aupr': avg_aupr,
            'fold_results': fold_results
        })
        all_auroc.append(avg_auroc)
        all_aupr.append(avg_aupr)
        
        # 记录当前实验结果到文件
        results_path = os.path.join(exp_dir, "experiment_results.csv")
        with open(results_path, 'w') as f:
            f.write("fold,auroc,aupr\n")
            for i, (fold_auroc, fold_aupr) in enumerate(fold_results):
                f.write(f"{i+1},{fold_auroc:.4f},{fold_aupr:.4f}\n")
            f.write(f"average,{avg_auroc:.4f},{avg_aupr:.4f}\n")
        
        print(f"Experiment {exp_idx+1} (Seed {seed}) - Avg AUROC: {avg_auroc:.4f}, Avg AUPR: {avg_aupr:.4f}")
    
    # 计算所有实验的综合统计数据
    overall_avg_auroc = sum(all_auroc) / len(all_auroc)
    overall_avg_aupr = sum(all_aupr) / len(all_aupr)
    
    # 计算标准差
    auroc_std = np.std(all_auroc)
    aupr_std = np.std(all_aupr)
    
    # 找出最佳和最差的实验
    best_exp_idx = np.argmax(all_auroc)
    worst_exp_idx = np.argmin(all_auroc)
    
    # 输出综合统计结果
    print("\n===== OVERALL RESULTS =====")
    print(f"Overall Average - AUROC: {overall_avg_auroc:.4f} ± {auroc_std:.4f}, AUPR: {overall_avg_aupr:.4f} ± {aupr_std:.4f}")
    print(f"Best Result (Seed {fixed_seeds[best_exp_idx]}) - AUROC: {all_auroc[best_exp_idx]:.4f}, AUPR: {all_aupr[best_exp_idx]:.4f}")
    print(f"Worst Result (Seed {fixed_seeds[worst_exp_idx]}) - AUROC: {all_auroc[worst_exp_idx]:.4f}, AUPR: {all_aupr[worst_exp_idx]:.4f}")
    
    # 保存所有实验的综合结果
    summary_path = os.path.join("seed_experiments", "summary_results.csv")
    with open(summary_path, 'w') as f:
        f.write("experiment,seed,avg_auroc,avg_aupr\n")
        for i, res in enumerate(all_results):
            f.write(f"{i+1},{res['seed']},{res['avg_auroc']:.4f},{res['avg_aupr']:.4f}\n")
        f.write(f"overall,NA,{overall_avg_auroc:.4f},{overall_avg_aupr:.4f}\n")
        f.write(f"std,NA,{auroc_std:.4f},{aupr_std:.4f}\n")