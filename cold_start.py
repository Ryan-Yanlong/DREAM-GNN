#!/usr/bin/env python3
"""
Gdataset冷启动实验脚本 - 无信息泄露版本
专门用于测试在Gdataset上训练的模型

核心功能：
1. 加载训练好的药物和疾病嵌入向量
2. 使用原始drug_embed计算相似度（无信息泄露）
3. 支持疾病条件化聚合
4. 测试不同K值的效果
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
    Gdataset冷启动实验实现
    无信息泄露，使用训练好的嵌入向量
    """
    
    def __init__(self, device=None, disease_conditional_config=None):
        self.device = device if device else th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # 疾病条件化聚合的配置
        if disease_conditional_config is None:
            disease_conditional_config = {
                'alpha': 0.7,  # 疾病匹配度权重
                'beta': 0.3,   # 药物相似度权重
                'temperature': 0.1,  # softmax温度参数
                'use_disease_aware': True
            }
        self.disease_conditional_config = disease_conditional_config
        
        print(f"Gdataset冷启动实验 - 无信息泄露版本")
        print(f"设备: {self.device}")
        print(f"疾病条件化配置: {disease_conditional_config}")
        
        # 加载原始Gdataset数据
        self.load_raw_data()
    
    def load_raw_data(self):
        """加载原始Gdataset数据"""
        data_path = './raw_data/drug_data/Gdataset/Gdataset.mat'
        
        if not os.path.exists(data_path):
            print(f"错误: 找不到Gdataset数据文件: {data_path}")
            return False
        
        try:
            data = sio.loadmat(data_path)
            
            self.association_matrix = data['didr'].T  # 药物-疾病关联矩阵
            self.drug_embed_raw = data['drug_embed']  # 原始药物嵌入
            self.disease_embed_raw = data['disease_embed']  # 原始疾病嵌入
            
            self.num_drug = self.association_matrix.shape[0]
            self.num_disease = self.association_matrix.shape[1]
            
            print(f"✓ Gdataset数据加载完成:")
            print(f"  药物数: {self.num_drug}")
            print(f"  疾病数: {self.num_disease}")
            print(f"  关联数: {np.sum(self.association_matrix == 1)}")
            print(f"  正样本率: {np.sum(self.association_matrix == 1) / (self.num_drug * self.num_disease):.4f}")
            
            return True
            
        except Exception as e:
            print(f"错误: 加载Gdataset数据失败: {e}")
            return False
    
    def load_fold_data(self, fold, model_dir='seed_experiments/seed_77'):
        """加载指定折的训练数据和嵌入向量"""
        print(f"\n加载第{fold}折数据...")
        
        # 检查文件路径
        model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
        embeddings_path = os.path.join(model_dir, f"embeddings_fold{fold}.pth")
        metadata_path = os.path.join(model_dir, f"cold_start_metadata_fold{fold}.json")
        
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件: {model_path}")
            return False
            
        if not os.path.exists(embeddings_path):
            print(f"错误: 找不到嵌入文件: {embeddings_path}")
            return False
        
        try:
            # 加载MLP decoder权重
            state_dict = th.load(model_path, map_location=self.device, weights_only=False)
            
            # 重建MLP decoder (与训练时保持一致)
            self.mlp_decoder = nn.Sequential(
                nn.Linear(256, 128),  # 2 * 128 = 256 (drug_feat + dis_feat)
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)
            
            # 加载权重
            self.mlp_decoder[0].weight.data = state_dict['decoder.lin1.weight'].to(self.device)
            self.mlp_decoder[0].bias.data = state_dict['decoder.lin1.bias'].to(self.device)
            self.mlp_decoder[2].weight.data = state_dict['decoder.lin2.weight'].to(self.device)
            self.mlp_decoder[2].bias.data = state_dict['decoder.lin2.bias'].to(self.device)
            self.mlp_decoder[4].weight.data = state_dict['decoder.lin3.weight'].to(self.device)
            self.mlp_decoder[4].bias.data = state_dict['decoder.lin3.bias'].to(self.device)
            
            self.mlp_decoder.eval()
            
            # 加载嵌入数据
            embeddings_data = th.load(embeddings_path, map_location=self.device)
            
            # 关键：确保只使用训练集药物
            self.train_drug_indices = embeddings_data['train_drug_indices']
            self.train_disease_indices = embeddings_data['train_disease_indices']
            
            # 加载训练好的嵌入
            if 'drug_feats' in embeddings_data:
                self.drug_embeddings_trained = embeddings_data['drug_feats'].to(self.device)
                self.disease_embeddings_trained = embeddings_data['dis_feats'].to(self.device)
            else:
                self.drug_embeddings_trained = embeddings_data['drug_out'].to(self.device)
                self.disease_embeddings_trained = embeddings_data['dis_out'].to(self.device)
            
            # 加载元数据（如果存在）
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✓ 元数据加载成功")
            else:
                self.metadata = {}
            
            print(f"✓ 数据加载成功")
            print(f"  训练集药物数: {len(self.train_drug_indices)}")
            print(f"  训练集疾病数: {len(self.train_disease_indices)}")
            print(f"  训练集药物嵌入维度: {self.drug_embeddings_trained.shape}")
            print(f"  训练集疾病嵌入维度: {self.disease_embeddings_trained.shape}")
            
            return True
            
        except Exception as e:
            print(f"错误: 加载折数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_raw_similarity(self, query_drug_idx, candidate_drug_indices):
        """基于原始drug_embed计算相似度 - 无信息泄露"""
        # 获取查询药物的原始嵌入
        query_embed = self.drug_embed_raw[query_drug_idx]
        
        # 获取候选药物的原始嵌入
        candidate_embeds = self.drug_embed_raw[candidate_drug_indices]
        
        # 计算余弦相似度
        similarities = []
        for candidate_embed in candidate_embeds:
            # 归一化
            query_norm = np.linalg.norm(query_embed)
            candidate_norm = np.linalg.norm(candidate_embed)
            
            if query_norm > 0 and candidate_norm > 0:
                sim = np.dot(query_embed, candidate_embed) / (query_norm * candidate_norm)
            else:
                sim = 0.0
            
            similarities.append(sim)
        
        return np.array(similarities)
    
    def aggregate_drug_embeddings(self, query_drug_idx, candidate_drug_indices, k_values=[3, 5, 10, 15, 20]):
        """聚合药物嵌入 - 测试不同K值的效果"""
        results = {}
        
        # 计算原始相似度
        raw_similarities = self.compute_raw_similarity(query_drug_idx, candidate_drug_indices)
        
        # 测试不同K值
        for k in k_values:
            if k > len(candidate_drug_indices):
                continue
            
            # 选择Top-K候选药物
            top_k_indices = np.argsort(-raw_similarities)[:k]
            top_k_candidate_indices = candidate_drug_indices[top_k_indices]
            top_k_similarities = raw_similarities[top_k_indices]
            
            # 获取对应的训练好的药物嵌入
            top_k_embeddings = self.drug_embeddings_trained[top_k_candidate_indices]
            
            # 计算权重 (softmax)
            weights = F.softmax(th.tensor(top_k_similarities, dtype=th.float32, device=self.device) / 0.1, dim=0)
            
            # 加权聚合
            aggregated_drug_embedding = th.sum(weights.unsqueeze(-1) * top_k_embeddings, dim=0)
            
            # 记录结果
            results[k] = {
                'aggregated_embedding': aggregated_drug_embedding,
                'weights': weights.cpu().numpy(),
                'top_k_indices': top_k_candidate_indices,
                'top_k_similarities': top_k_similarities,
                'max_similarity': np.max(top_k_similarities),
                'avg_similarity': np.mean(top_k_similarities)
            }
        
        return results
    
    def predict_associations(self, drug_embedding, disease_indices):
        """预测药物-疾病关联 - 保持与训练过程一致"""
        with th.no_grad():
            # 获取疾病嵌入 (训练好的)
            disease_embeddings = self.disease_embeddings_trained[disease_indices]
            
            # 扩展药物嵌入以匹配疾病数量
            drug_expanded = drug_embedding.unsqueeze(0).expand(len(disease_indices), -1)
            
            # 拼接药物和疾病特征 (与训练时保持一致)
            combined_features = th.cat([drug_expanded, disease_embeddings], dim=1)
            
            # 通过MLP decoder
            logits = self.mlp_decoder(combined_features).squeeze(-1)
            
            # 应用sigmoid得到概率
            probabilities = th.sigmoid(logits).cpu().numpy()
        
        return probabilities
    
    def evaluate_cold_start(self, num_test_drugs=50, k_values=[3, 5, 10, 15, 20], random_seed=42):
        """评估冷启动性能"""
        np.random.seed(random_seed)
        
        print(f"\n=== 冷启动评估 ===")
        print(f"测试药物数: {num_test_drugs}")
        print(f"测试K值: {k_values}")
        
        # 随机选择测试药物
        all_drug_indices = np.arange(self.num_drug)
        test_drug_indices = np.random.choice(all_drug_indices, num_test_drugs, replace=False)
        
        # 确保候选药物来自训练集
        candidate_drug_indices = np.setdiff1d(self.train_drug_indices, test_drug_indices)
        
        print(f"  测试药物数: {len(test_drug_indices)}")
        print(f"  候选药物数: {len(candidate_drug_indices)}")
        
        # 记录不同K值的结果
        k_results = {k: {'predictions': [], 'labels': [], 'drug_details': []} for k in k_values}
        
        for drug_idx in tqdm(test_drug_indices, desc="评估药物"):
            # 获取真实标签
            true_associations = self.association_matrix[drug_idx]
            true_positive_diseases = np.where(true_associations == 1)[0]
            num_positives = len(true_positive_diseases)
            
            if num_positives == 0:
                continue
            
            # 聚合药物嵌入 (测试不同K值)
            aggregation_results = self.aggregate_drug_embeddings(drug_idx, candidate_drug_indices, k_values)
            
            # 为每个K值进行预测
            for k, agg_result in aggregation_results.items():
                # 预测所有疾病
                predictions = self.predict_associations(agg_result['aggregated_embedding'], np.arange(self.num_disease))
                
                # 记录结果
                k_results[k]['predictions'].extend(predictions)
                k_results[k]['labels'].extend(true_associations)
                
                # 记录药物详细信息
                drug_detail = {
                    'drug_idx': drug_idx,
                    'num_positives': num_positives,
                    'max_similarity': agg_result['max_similarity'],
                    'avg_similarity': agg_result['avg_similarity'],
                    'k_used': k,
                    'positive_scores': predictions[true_positive_diseases]
                }
                k_results[k]['drug_details'].append(drug_detail)
        
        # 计算每个K值的性能
        final_results = {}
        for k in k_values:
            if len(k_results[k]['predictions']) > 0:
                predictions = np.array(k_results[k]['predictions'])
                labels = np.array(k_results[k]['labels'])
                
                # 计算指标
                auroc = roc_auc_score(labels, predictions)
                aupr = average_precision_score(labels, predictions)
                
                # 不同阈值的性能
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
    
    def analyze_results(self, results):
        """分析结果"""
        print(f"\n=== 结果分析 ===")
        
        # 1. 整体性能对比
        print(f"不同K值的性能对比:")
        print(f"{'K值':<6} {'AUROC':<8} {'AUPR':<8} {'阈值0.3 F1':<12}")
        print("-" * 40)
        
        for k in sorted(results.keys()):
            auroc = results[k]['auroc']
            aupr = results[k]['aupr']
            f1_03 = results[k]['threshold_metrics'][0.3]['f1']
            print(f"{k:<6} {auroc:<8.4f} {aupr:<8.4f} {f1_03:<12.4f}")
        
        # 2. 相似度分析
        print(f"\n相似度分析:")
        for k in sorted(results.keys()):
            drug_details = results[k]['drug_details']
            max_sims = [detail['max_similarity'] for detail in drug_details]
            avg_sims = [detail['avg_similarity'] for detail in drug_details]
            
            print(f"K={k}: 最大相似度={np.mean(max_sims):.4f}±{np.std(max_sims):.4f}, "
                  f"平均相似度={np.mean(avg_sims):.4f}±{np.std(avg_sims):.4f}")
        
        # 3. 正样本恢复分析
        print(f"\n正样本恢复分析 (阈值0.3):")
        for k in sorted(results.keys()):
            drug_details = results[k]['drug_details']
            recovery_rates = []
            
            for detail in drug_details:
                positive_scores = detail['positive_scores']
                recovered = np.sum(positive_scores >= 0.3)
                recovery_rate = recovered / detail['num_positives']
                recovery_rates.append(recovery_rate)
            
            if recovery_rates:
                avg_recovery = np.mean(recovery_rates)
                print(f"K={k}: 平均恢复率={avg_recovery:.4f} (n={len(recovery_rates)})")
    
    def run_experiment(self, cv_folds=3, model_dir='seed_experiments/seed_77'):
        """运行完整实验"""
        all_results = []
        
        for fold in range(1, cv_folds + 1):
            print(f"\n{'='*60}")
            print(f"第 {fold} 折实验")
            print(f"{'='*60}")
            
            try:
                if not self.load_fold_data(fold, model_dir):
                    continue
                
                # 运行冷启动评估
                results = self.evaluate_cold_start(
                    num_test_drugs=50,
                    k_values=[3, 5, 10, 15, 20],
                    random_seed=42 + fold
                )
                
                # 分析结果
                self.analyze_results(results)
                all_results.append(results)
                
            except Exception as e:
                print(f"第{fold}折实验失败: {e}")
                import traceback
                traceback.print_exc()
        
        return all_results


def main():
    """主函数"""
    print(f"Gdataset冷启动实验")
    print(f"{'='*80}")
    
    # 初始化冷启动实验
    cold_start = GdatasetColdStart()
    
    # 检查数据加载
    if not hasattr(cold_start, 'association_matrix'):
        print("错误: 数据加载失败，请检查Gdataset.mat文件")
        return
    
    # 运行实验
    print(f"\n开始运行冷启动实验...")
    results = cold_start.run_experiment(
        cv_folds=3,  # 测试前3折
        model_dir='seed_experiments/seed_77'  # 使用你的模型目录
    )
    
    print(f"\n实验完成！")
    print(f"共完成 {len(results)} 折实验")


if __name__ == "__main__":
    main()
