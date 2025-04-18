import os
import argparse
import numpy as np
import torch as th
from model import Net
from data import DrugDataLoader
from utils import setup_seed
from drug_train import train, get_top_novel_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaDR with Data Augmentation and L2 Regularization')
    parser.add_argument('--device', default='0', type=int,
                        help='运行设备，例如 "--device 0"，若使用 CPU 则设置为 --device -1')
    parser.add_argument('--save_dir', type=str, default='seed_0_results', help='日志保存目录')
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
    parser.add_argument('--data_name', default='Cdataset', type=str)
    parser.add_argument('--num_neighbor', type=int, default=4, help='default number of neighbors')
    parser.add_argument('--beta', type=float, default=0.001, help='公共表示学习损失权重')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='优化器权重衰减系数')
    parser.add_argument('--l2_reg_weight', type=float, default=0.0000, help='L2正则化权重')
    parser.add_argument('--attention_dropout', type=float, default=0.1, 
                        help='注意力机制中的dropout率')
    parser.add_argument('--embedding_mode', type=str, default='pretrained', choices=['pretrained', 'random'],
                        help='选择使用预训练embedding还是随机初始化')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='是否使用数据增强')
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
    parser.add_argument('--save_model', action='store_true', default=True, help='是否保存最佳模型')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑程度，0表示不使用')
    parser.add_argument('--generate_top_predictions', action='store_true', default=True, 
                        help='训练后生成顶部新型预测')
    parser.add_argument('--top_k', type=int, default=200, 
                        help='要生成的顶部预测数量')
    
    parser.set_defaults(use_gate_attention=False)

    args = parser.parse_args()
    print(args)

    # 只使用seed 0
    seed = 0
    print(f"======== Running with seed {seed} ========")
    
    # 设置随机种子
    setup_seed(seed)
    
    # 创建保存目录
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 设置设备
    if args.device >= 0:
        args.device = f"cuda:{args.device}" if th.cuda.is_available() else "cpu"
    else:
        args.device = "cpu"
    print(f"Using device: {args.device}")

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

    # 存储所有折的top预测结果
    all_top_predictions = []
    
    # 在每个折上训练并保存top预测
    for cv in range(10):
        args.save_id = cv + 1
        print("============== Fold {} ==============".format(cv + 1))
        
        # 训练模型
        auroc, aupr = train(args, dataset, cv)
        print(f"Fold {cv+1} - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
        
        # 加载最佳模型并生成top预测
        if args.save_model and args.generate_top_predictions:
            print(f"\n为第{cv+1}折生成top {args.top_k}预测...")
            best_model = Net(args=args).to(args.device)
            model_path = os.path.join(args.save_dir, f"best_model_fold{args.save_id}.pth")
            
            if os.path.exists(model_path):
                best_model.load_state_dict(th.load(model_path))
                top_predictions = get_top_novel_predictions(args, best_model, dataset, cv, top_k=args.top_k)
                
                # 添加折索引列
                top_predictions['fold'] = cv + 1
                all_top_predictions.append(top_predictions)
                
                print(f"第{cv+1}折的前5个预测:\n{top_predictions.head(5)}")
            else:
                print(f"警告: 未找到第{cv+1}折的模型文件 {model_path}")
    
    # 合并所有折的预测并按分数排序
    if all_top_predictions:
        import pandas as pd
        combined_predictions = pd.concat(all_top_predictions, ignore_index=True)
        combined_predictions = combined_predictions.sort_values('score', ascending=False).reset_index(drop=True)
        
        # 保存所有折的合并预测
        combined_path = os.path.join(args.save_dir, "combined_top_predictions.csv")
        combined_predictions.to_csv(combined_path, index=False)
        
        # 输出整体top预测
        print("\n所有折综合的前10个预测:")
        for i, row in combined_predictions.head(10).iterrows():
            drug_id = row['drug_id']
            disease_id = row['disease_id']
            drug_name = row.get('drug_name', f"Drug_{drug_id}")
            print(f"排名 {i+1}: 药物ID {drug_id} ({drug_name}) - 疾病ID {disease_id}, 得分: {row['score']:.4f}, 来自第{row['fold']}折")
    else:
        print("警告: 未生成任何预测结果")
