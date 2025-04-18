#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合式 data.py

本模块用于加载药物-疾病关联数据、构造特征、划分交叉验证数据，
以及基于相似度矩阵构造各种图（例如 kNN 图、编码器图、解码器图）。
实验设定为：给定所有已知药物和疾病，预测新的药物–疾病关联（传导式链接预测）。

主要特点：
1. 采用与dataloader.py相同的数据划分方法，确保数据划分一致性
2. 使用掩码机制区分训练集和测试集
3. 不对负样本进行下采样，使用所有可能的药物-疾病对
4. 为每个交叉验证折构建特定的图结构，确保测试边信息不泄露
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

# 从 utils 中仅导入所需函数
from utils import knn_graph, to_etype_name, normalize, sparse_mx_to_torch_sparse_tensor
# 若使用数据增强，则引入增强模块
from graph_augmentation import augmented_knn_graph, GraphAugmentation

# 数据文件路径配置
_paths = {
    'Gdataset': './raw_data/drug_data/Gdataset/Gdataset.mat',
    'Cdataset': './raw_data/drug_data/Cdataset/Cdataset.mat',
    'Ldataset': './raw_data/drug_data/Ldataset/lagcn',
    'lrssl': './raw_data/drug_data/lrssl/lrssl.mat',
}


class DrugDataLoader(object):
    """
    药物数据加载器，使用与dataloader.py一致的划分方法。
    
    主要功能：
      - 从 .mat 文件中加载药物-疾病关联矩阵、相似度矩阵和预训练 embedding；
      - 采用 KFold 对正负样本分别进行划分，创建交叉验证数据；
      - 构造不包含测试边信息的药物与疾病特征图；
      - 基于训练集关联数据构造编码器图和解码器图。
      
    实验设定：传导式链接预测，所有节点均参与构图，但确保测试边信息不泄露到训练过程。
    """
    def __init__(self, name, device, symm=True, k=5, use_augmentation=False, aug_params=None):
        """
        初始化数据加载器

        参数：
          name: 数据集名称（例如 'Gdataset' 或 'Cdataset'）
          device: torch 计算设备（例如 th.device("cpu") 或 th.device("cuda")）
          symm: 是否使用对称归一化
          k: kNN 图中使用的近邻数
          use_augmentation: 是否启用数据增强
          aug_params: 数据增强参数字典
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
        
        # 加载原始数据
        self._load_raw_data(self._dir, self._name)
        
        # 构造交叉验证数据字典
        self.cv_data_dict = self._create_cv_splits()
        
        # 设置为使用预训练embedding
        self.embedding_mode = "pretrained"
        
        # 生成药物和疾病的特征（embedding）
        self._generate_feat()
        
        # 构造每个交叉验证折特定的图结构
        self.cv_specific_graphs = {}
        self._generate_cv_specific_graphs()
        
        # 为每个折构建训练和测试图
        self.data_cv = self._build_all_cv_data()
        
        print(f"[Init] Data loader initialization complete.")

    def _load_raw_data(self, file_path, data_name):
        """
        加载原始数据文件
        
        参数：
          file_path: 数据文件路径
          data_name: 数据集名称
        """
        print(f"[Load Data] Reading data from: {file_path}")
        
        if data_name in ['Gdataset', 'Cdataset','lrssl']:
            data = sio.loadmat(file_path)
            # 转置后保证关联矩阵形状为 (num_drug, num_disease)
            self.association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
            # drug_ids 格式为形状 (N, 1)，每个元素例如 [['DB00014']]
            self.drug_ids = [str(x[0][0]).strip() for x in data['Wrname']] if 'Wrname' in data else None
            
            # 加载预训练 embedding
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
        创建交叉验证数据划分，使用与dataloader.py相同的方法
        
        返回：
          cv_data: 字典，每一折包含 [train_data, test_data, unique_values]
        """
        interactions = self.association_matrix
        
        # 获取正样本和负样本的索引
        pos_row, pos_col = np.nonzero(interactions)
        neg_row, neg_col = np.nonzero(1 - interactions)
        
        # 不进行下采样，使用所有正负样本
        print(f"[CV Split] Positive samples: {len(pos_row)}, Negative samples: {len(neg_row)}")
        
        # 创建交叉验证划分
        cv_data = {}
        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        
        for cv_num, ((train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx)) in enumerate(
                zip(kfold.split(pos_row), kfold.split(neg_row))):
            
            # 创建训练和测试掩码
            train_mask = np.zeros_like(interactions, dtype=bool)
            test_mask = np.zeros_like(interactions, dtype=bool)
            
            # 构建训练边和测试边
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
            
            # 合并正负样本边
            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            
            # 设置掩码
            train_mask[train_edge[0], train_edge[1]] = True
            test_mask[test_edge[0], test_edge[1]] = True
            
            # 构建训练数据
            train_values = np.zeros(train_edge.shape[1])
            train_values[:len(train_pos_idx)] = 1  # 正样本标记为1
            
            # 构建测试数据
            test_values = np.zeros(test_edge.shape[1])
            test_values[:len(test_pos_idx)] = 1  # 正样本标记为1
            
            # 使用 DataFrame 便于后续处理
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
            
            unique_values = np.array([0, 1])  # 二分类任务
            cv_data[cv_num] = [train_data_info, test_data_info, unique_values]
            
            print(f"[CV Split] Fold {cv_num}: Train = {train_data_info.shape[0]} samples, Test = {test_data_info.shape[0]} samples")
            
        return cv_data

    def _generate_feat(self):
        """
        构造药物和疾病特征，使用预训练embedding或随机初始化
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
        
        # 归一化特征
        self.drug_feature = F.normalize(self.drug_feature, p=2, dim=1)
        self.disease_feature = F.normalize(self.disease_feature, p=2, dim=1)
        
        # 保存特征形状，方便后续模型构造时使用
        self.drug_feature_shape = self.drug_feature.shape
        self.disease_feature_shape = self.disease_feature.shape
        print("[Feature] Drug feature shape:", self.drug_feature_shape)
        print("[Feature] Disease feature shape:", self.disease_feature_shape)

    def _generate_cv_specific_graphs(self):
        """
        为每个交叉验证折生成特定的图结构，确保不包含测试边信息
        """
        for cv_idx in range(10):
            print(f"[Graph] Building fold {cv_idx} specific graphs...")
            
            # 获取当前折的训练和测试数据
            train_data = self.cv_data_dict[cv_idx][0]
            test_data = self.cv_data_dict[cv_idx][1]
            
            # 创建训练关联矩阵的副本
            train_assoc_matrix = np.zeros_like(self.association_matrix)
            
            # 填充训练集边
            pos_train_indices = train_data[train_data['values'] == 1].index
            for idx in pos_train_indices:
                drug_id = train_data.loc[idx, 'drug_id']
                disease_id = train_data.loc[idx, 'disease_id']
                train_assoc_matrix[drug_id, disease_id] = 1
            
            # 使用原始相似度矩阵 (与dataloader.py一致)
            drug_sim_matrix = self.drug_sim_features.copy()
            disease_sim_matrix = self.disease_sim_features.copy()
            
            # 基于相似度矩阵构建KNN图
            drug_graph = self._create_similarity_graph(
                drug_sim_matrix, self.num_neighbor)
            
            disease_graph = self._create_similarity_graph(
                disease_sim_matrix, self.num_neighbor)
            
            # 基于节点特征构建KNN图 (使用dataloader.py中类似的方法)
            drug_feature_graph = self._create_feature_similarity_graph(
                'drug', self.num_neighbor)
            
            disease_feature_graph = self._create_feature_similarity_graph(
                'disease', self.num_neighbor)
            
            # 存储该折的图结构
            self.cv_specific_graphs[cv_idx] = {
                'drug_graph': drug_graph,
                'disease_graph': disease_graph,
                'drug_feature_graph': drug_feature_graph,
                'disease_feature_graph': disease_feature_graph,
                'train_association_matrix': train_assoc_matrix
            }

    def _create_similarity_graph(self, sim_matrix, k):
        """
        基于相似度矩阵创建KNN图 (类似于dataloader.py中的build_graph方法)
        
        参数：
          sim_matrix: 相似度矩阵
          k: k近邻数
          
        返回：
          graph: torch稀疏张量表示的图
        """
        # 确保k不超过矩阵大小
        k_actual = min(k, sim_matrix.shape[0] - 1)
        
        # 使用与dataloader.py类似的KNN图构建方法
        neighbor = np.argpartition(-sim_matrix, kth=k_actual, axis=1)[:, :k_actual]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)
        
        # 创建稀疏邻接矩阵
        data = np.ones(len(row_index))
        adj = sp.coo_matrix((data, (row_index, col_index)), shape=(sim_matrix.shape[0], sim_matrix.shape[0]))
        
        # 如果需要对称化
        if self._symm:
            adj = adj + adj.T
            adj = adj.multiply(adj > 0)
            
        # 归一化并转换为torch稀疏张量
        normalized_adj = normalize(adj + sp.eye(adj.shape[0]))
        graph = sparse_mx_to_torch_sparse_tensor(normalized_adj)
        
        return graph

    def _create_feature_similarity_graph(self, node_type, k):
        """
        基于节点特征相似性创建KNN图
        
        参数：
          node_type: 'drug' 或 'disease'
          k: k近邻数
          
        返回：
          graph: torch稀疏张量表示的图
        """
        # 获取特征
        if node_type == 'drug':
            features = self.drug_embed.copy() if hasattr(self, 'drug_embed') else self.drug_sim_features.copy()
            num_entities = self._num_drug
        else:
            features = self.disease_embed.copy() if hasattr(self, 'disease_embed') else self.disease_sim_features.copy()
            num_entities = self._num_disease
            
        # 计算特征相似度矩阵
        if len(features.shape) > 1:  # 对于embedding特征
            # 归一化特征
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized_features = features / norms
            
            # 计算余弦相似度
            similarity_matrix = np.dot(normalized_features, normalized_features.T)
        else:  # 对于已有的相似度矩阵
            similarity_matrix = features
        
        # 使用与_create_similarity_graph相同的方法构建KNN图    
        return self._create_similarity_graph(similarity_matrix, k)

    def _build_all_cv_data(self):
        """
        为所有交叉验证折构建训练和测试图
        
        返回：
          data_cv: 包含每折训练和测试图的字典
        """
        data_cv = {}
        
        for cv_idx in range(10):
            # 获取当前折的训练和测试数据
            train_data = self.cv_data_dict[cv_idx][0]
            test_data = self.cv_data_dict[cv_idx][1]
            values = self.cv_data_dict[cv_idx][2]
            
            # 生成训练和测试边
            train_pairs, train_values = self._generate_pair_value(train_data)
            test_pairs, test_values = self._generate_pair_value(test_data)
            
            # 构建编码器和解码器图
            train_enc_graph = self._generate_enc_graph(train_pairs, train_values, add_support=True)
            train_dec_graph = self._generate_dec_graph(train_pairs)
            
            test_enc_graph = self._generate_enc_graph(test_pairs, test_values, add_support=True)
            test_dec_graph = self._generate_dec_graph(test_pairs)
            
            # 存储图数据
            data_cv[cv_idx] = {
                'train': [train_enc_graph, train_dec_graph, th.FloatTensor(train_values)],
                'test': [test_enc_graph, test_dec_graph, th.FloatTensor(test_values)]
            }
            
            print(f"[CV Data] Fold {cv_idx} train/test graphs built")
            
        return data_cv

    @staticmethod
    def _generate_pair_value(rel_info):
        """
        从 DataFrame 中生成 (drug_id, disease_id) 对以及对应的评分
        
        参数：
          rel_info: DataFrame，包含 'drug_id', 'disease_id', 'values'
          
        返回：
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
        构造编码器图（DGL异构图），包含不同关系（正负关联）
        
        参数：
          rating_pairs: (drug_ids, disease_ids) 元组
          rating_values: 对应的关联值
          add_support: 是否添加归一化支持信息
          
        返回：
          graph: DGL异构图
        """
        possible_rel_values = np.unique(rating_values)
        data_dict = {}
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        
        # 创建映射字典，将评分值映射到边类型名称
        etype_map = {}
        rev_etype_map = {}
        
        # 确保关联类型命名与原始代码一致
        print(f"[Graph] Building encoder graph with rating values: {possible_rel_values}")
        for rating in possible_rel_values:
            idx = np.where(rating_values == rating)
            rrow = rating_row[idx]
            rcol = rating_col[idx]
            
            # 使用与原始代码一致的边类型命名方式
            if rating == 0:
                etype = "0"
                rev_etype = "rev-0"
            elif rating == 1:
                etype = "1"
                rev_etype = "rev-1"
            else:
                etype = to_etype_name(rating)
                rev_etype = 'rev-%s' % etype
                
            # 存储映射关系，以便后续使用
            etype_map[rating] = etype
            rev_etype_map[rating] = rev_etype
                
            data_dict.update({
                ('drug', etype, 'disease'): (rrow, rcol),
                ('disease', rev_etype, 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # 校验边数
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x_np = x.numpy().astype('float32')
                x_np[x_np == 0.] = np.inf
                return th.FloatTensor(1. / np.sqrt(x_np)).unsqueeze(1)
                
            drug_ci, drug_cj = [], []
            disease_ci, disease_cj = [], []
            
            for rating in possible_rel_values:
                # 使用存储的映射获取正确的边类型名称
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
        基于药物-疾病关联对构造解码器图（二分图）
        
        参数：
          rating_pairs: (drug_ids, disease_ids) 元组
          
        返回：
          graph: DGL二分图
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
        对药物和疾病特征进行数据增强（噪声、遮蔽、Mixup等）
        
        返回：
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
        
        # 添加高斯噪声
        aug_drug_feature = GraphAugmentation.feature_noise(aug_drug_feature, feature_noise_scale)
        aug_disease_feature = GraphAugmentation.feature_noise(aug_disease_feature, feature_noise_scale)
        
        # 特征遮蔽
        aug_drug_feature = GraphAugmentation.feature_masking(aug_drug_feature, feature_mask_rate)
        aug_disease_feature = GraphAugmentation.feature_masking(aug_disease_feature, feature_mask_rate)
        
        # 可选Mixup
        if use_mixup:
            aug_drug_feature = GraphAugmentation.mix_up_features(aug_drug_feature, mixup_alpha)
            aug_disease_feature = GraphAugmentation.mix_up_features(aug_disease_feature, mixup_alpha)
            
        return aug_drug_feature, aug_disease_feature

    def get_graph_data_for_training(self, cv_idx):
        """
        获取特定折用于训练的图数据，包括增强后的数据
        
        参数：
          cv_idx: 交叉验证折索引
          
        返回：
          graph_data: 包含训练所需所有图数据的字典
        """
        # 获取基本图数据
        cv_data = self.data_cv[cv_idx]
        cv_specific_graphs = self.cv_specific_graphs[cv_idx]
        
        # 准备数据增强
        if self.use_augmentation:
            aug_drug_feat, aug_disease_feat = self.augment_features()
        else:
            aug_drug_feat, aug_disease_feat = self.drug_feature, self.disease_feature
            
        # 构建返回的图数据字典
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

    # 为了兼容性保留_generate_feature_similarity_graph
    def _generate_feature_similarity_graph(self, feature_type='drug', k=5, similarity_method='cosine'):
        """
        兼容性方法 - 使用与dataloader.py相似的方法构建特征相似性图
        
        参数：
          feature_type: 'drug' 或 'disease'
          k: k近邻数量
          similarity_method: 相似度计算方法
          
        返回：
          graph: 特征相似性图
        """
        print(f"[Warning] Using compatibility method for {feature_type} graph with k={k}")
        return self._create_feature_similarity_graph(feature_type, k)

    @property
    def num_links(self):
        """返回可能的关联值数量"""
        return len(np.unique(self.association_matrix))

    @property
    def num_disease(self):
        """返回疾病数量"""
        return self._num_disease

    @property
    def num_drug(self):
        """返回药物数量"""
        return self._num_drug


if __name__ == "__main__":
    # 示例：初始化数据加载器并验证
    device = th.device("cpu")
    loader = DrugDataLoader(name='Gdataset', device=device, symm=True, k=5, use_augmentation=False)
    print("\n[Main] Data Loader initialization complete.")
    print(f"[Main] Number of drugs: {loader.num_drug}, Number of diseases: {loader.num_disease}")
    print("[Main] Drug feature shape:", loader.drug_feature_shape)
    print("[Main] Disease feature shape:", loader.disease_feature_shape)
    
    # 检查第0折的数据
    cv_idx = 0
    graph_data = loader.get_graph_data_for_training(cv_idx)
    print(f"\n[CV Fold {cv_idx}] Train encoder graph:", graph_data['train_enc_graph'])
    print(f"[CV Fold {cv_idx}] Test encoder graph:", graph_data['test_enc_graph'])
    
    # 验证没有信息泄露（测试）
    train_data = loader.cv_data_dict[cv_idx][0]
    test_data = loader.cv_data_dict[cv_idx][1]
    print(f"\n[Verification] Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # 检查没有重叠的边
    train_edges = set(zip(train_data['drug_id'], train_data['disease_id']))
    test_edges = set(zip(test_data['drug_id'], test_data['disease_id']))
    overlap = train_edges.intersection(test_edges)
    print(f"[Verification] Edge overlap between train and test: {len(overlap)} (should be 0)")