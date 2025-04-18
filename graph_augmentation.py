import torch as th
import numpy as np
import scipy.sparse as sp
import dgl
import random
from utils import sparse_mx_to_torch_sparse_tensor

class GraphAugmentation:
    """
    图数据增强模块：提供多种数据增强方法，用于减轻图神经网络过拟合
    """
    @staticmethod
    def random_edge_dropout(graph, dropout_rate=0.1):
        """
        随机丢弃边
        
        参数:
        graph: DGL图对象
        dropout_rate: 丢弃边的比例 (0-1)
        
        返回:
        augmented_graph: 增强后的图
        """
        if not isinstance(graph, dgl.DGLGraph):
            return graph
            
        # 创建一个新的异构图，保留原图的结构但只包含我们想要保留的边
        data_dict = {}
        num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
        
        # 确定图所在的设备
        device = graph.device
        
        # 处理每种边类型
        for etype in graph.canonical_etypes:
            src_ntype, rel_type, dst_ntype = etype
            
            # 获取该类型的边数量
            num_edges = graph.number_of_edges(etype)
            
            if num_edges == 0:
                # 如果没有边，添加一个空的边列表
                data_dict[etype] = (th.tensor([], dtype=th.int64, device=device), 
                                   th.tensor([], dtype=th.int64, device=device))
                continue
            
            # 计算要保留的边的数量
            num_keep = max(1, int(num_edges * (1 - dropout_rate)))  # 至少保留一条边
            
            # 随机选择要保留的边
            perm = th.randperm(num_edges, device=device)
            edges_to_keep = perm[:num_keep]
            
            # 获取所有源节点、目标节点
            src, dst = graph.edges(etype=etype)
            
            # 选择要保留的边
            src_keep = src[edges_to_keep]
            dst_keep = dst[edges_to_keep]
            
            # 将保留的边添加到新的数据字典中
            data_dict[etype] = (src_keep, dst_keep)
        
        # 创建新的异构图
        new_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        
        # 复制节点特征
        for ntype in graph.ntypes:
            for key, feat in graph.nodes[ntype].data.items():
                new_graph.nodes[ntype].data[key] = feat.clone()
        
        # 复制边特征 (如果有的话)
        for etype in graph.canonical_etypes:
            if new_graph.number_of_edges(etype) > 0:
                for key, feat in graph.edges[etype].data.items():
                    # 使用索引选择对应的边特征
                    src_type, rel_type, dst_type = etype
                    if graph.number_of_edges(etype) == new_graph.number_of_edges(etype):
                        # 如果边数量相同，直接复制
                        new_graph.edges[etype].data[key] = feat.clone()
                    else:
                        # 否则，需要选择对应的特征
                        try:
                            indices = perm[:num_keep]  # 使用之前选择的边的索引
                            new_graph.edges[etype].data[key] = feat[indices].clone()
                        except Exception as e:
                            print(f"Warning: Could not copy edge feature {key} for edge type {etype}: {e}")
        
        return new_graph

    @staticmethod
    def random_edge_dropout_sparse(sparse_graph, dropout_rate=0.1):
        """
        随机丢弃稀疏图中的边
        
        参数:
        sparse_graph: 稀疏张量表示的图
        dropout_rate: 丢弃边的比例 (0-1)
        
        返回:
        augmented_graph: 增强后的图
        """
        if not isinstance(sparse_graph, th.Tensor) or not sparse_graph.is_sparse:
            return sparse_graph
            
        # 获取稀疏张量的索引和值
        indices = sparse_graph._indices()
        values = sparse_graph._values()
        shape = sparse_graph.shape
        device = sparse_graph.device
        
        # 计算要保留的边的数量
        num_edges = values.size(0)
        num_keep = max(1, int(num_edges * (1 - dropout_rate)))  # 至少保留一条边
        
        # 随机选择要保留的边
        perm = th.randperm(num_edges, device=device)
        keep_indices = perm[:num_keep]
        
        # 创建新的稀疏张量
        new_indices = indices[:, keep_indices]
        new_values = values[keep_indices]
        
        return th.sparse_coo_tensor(new_indices, new_values, shape, device=device)

    @staticmethod
    def add_random_edges(graph, add_rate=0.05, self_loops=False):
        """
        随机添加边
        
        参数:
        graph: DGL图对象
        add_rate: 相对于原始边数添加的新边比例
        self_loops: 是否允许自环
        
        返回:
        augmented_graph: 增强后的图
        """
        if not isinstance(graph, dgl.DGLGraph):
            return graph
            
        # 创建图的深拷贝
        augmented_graph = graph.clone()
        
        # 确定图所在的设备
        device = augmented_graph.device
        
        # 对每种边类型分别处理
        for etype in augmented_graph.canonical_etypes:
            src_ntype, rel_type, dst_ntype = etype
            
            # 获取该类型的边数量
            num_edges = augmented_graph.number_of_edges(etype)
            if num_edges == 0:
                continue  # 跳过没有边的类型
                
            # 计算要添加的边的数量
            num_add = max(1, int(num_edges * add_rate))
                
            # 获取源节点类型和目标节点类型的节点数量
            num_src_nodes = augmented_graph.number_of_nodes(src_ntype)
            num_dst_nodes = augmented_graph.number_of_nodes(dst_ntype)
            
            if num_src_nodes == 0 or num_dst_nodes == 0:
                continue  # 跳过没有节点的类型
            
            # 获取现有边，用于避免添加已存在的边
            existing_edges = set()
            for i, (s, d) in enumerate(zip(*augmented_graph.edges(etype=etype))):
                existing_edges.add((s.item(), d.item()))
            
            # 随机生成新的边
            new_src = []
            new_dst = []
            attempts = 0
            max_attempts = num_add * 10  # 限制尝试次数，避免无限循环
            
            while len(new_src) < num_add and attempts < max_attempts:
                src_idx = random.randint(0, num_src_nodes - 1)
                dst_idx = random.randint(0, num_dst_nodes - 1)
                
                # 如果不允许自环且源节点类型与目标节点类型相同，检查是否形成自环
                if not self_loops and src_ntype == dst_ntype and src_idx == dst_idx:
                    attempts += 1
                    continue
                    
                # 检查边是否已存在
                if (src_idx, dst_idx) not in existing_edges and (src_idx, dst_idx) not in zip(new_src, new_dst):
                    new_src.append(src_idx)
                    new_dst.append(dst_idx)
                
                attempts += 1
            
            # 添加新边
            if new_src:
                try:
                    augmented_graph.add_edges(
                        th.tensor(new_src, dtype=th.int64, device=device),
                        th.tensor(new_dst, dtype=th.int64, device=device),
                        etype=etype
                    )
                except Exception as e:
                    print(f"Warning: Error adding edges for edge type {etype}: {e}")
        
        return augmented_graph

    @staticmethod
    def feature_noise(features, noise_scale=0.1):
        """
        向特征添加高斯噪声
        
        参数:
        features: 节点特征张量
        noise_scale: 噪声标准差
        
        返回:
        noisy_features: 添加噪声后的特征
        """
        if features is None:
            return None
            
        if isinstance(features, th.Tensor):
            # 获取特征的设备
            device = features.device
            
            # 生成与特征相同形状的高斯噪声，并放在同一设备上
            noise = th.randn_like(features, device=device) * noise_scale
            noisy_features = features + noise
            return noisy_features
        else:
            # 如果不是张量，尝试转换
            try:
                features_tensor = th.tensor(features, dtype=th.float32)
                device = th.device('cuda' if th.cuda.is_available() else 'cpu')
                features_tensor = features_tensor.to(device)
                noise = th.randn_like(features_tensor) * noise_scale
                noisy_features = features_tensor + noise
                return noisy_features
            except Exception as e:
                print(f"Warning: Failed to add noise to features: {e}")
                return features
    
    @staticmethod
    def sparse_graph_noise(graph, noise_scale=0.05):
        """
        向稀疏图添加噪声
        
        参数:
        graph: 稀疏图 (torch.sparse_coo_tensor)
        noise_scale: 噪声标准差
        
        返回:
        noisy_graph: 添加噪声后的图
        """
        if not isinstance(graph, th.Tensor) or not graph.is_sparse:
            return graph
            
        # 获取稀疏张量的索引和值
        indices = graph._indices()
        values = graph._values()
        shape = graph.shape
        device = graph.device
        
        # 为值添加噪声 (确保在同一设备上)
        noise = th.randn_like(values, device=device) * noise_scale
        noisy_values = values + noise
        
        # 确保值在合理范围内(例如，非负)
        noisy_values = th.clamp(noisy_values, min=0.0)
        
        # 创建新的稀疏张量
        noisy_graph = th.sparse_coo_tensor(indices, noisy_values, shape, device=device)
        return noisy_graph
    
    @staticmethod
    def feature_masking(features, mask_rate=0.1):
        """
        随机遮蔽特征中的部分元素
        
        参数:
        features: 节点特征张量
        mask_rate: 遮蔽率 (0-1)
        
        返回:
        masked_features: 遮蔽后的特征
        """
        if features is None:
            return None
            
        if isinstance(features, th.Tensor):
            # 获取特征的设备
            device = features.device
            
            # 创建一个掩码张量，在同一设备上
            mask = (th.rand_like(features, device=device) > mask_rate)
            # 应用掩码
            masked_features = features * mask
            return masked_features
        else:
            try:
                device = th.device('cuda' if th.cuda.is_available() else 'cpu')
                features_tensor = th.tensor(features, dtype=th.float32, device=device)
                mask = (th.rand_like(features_tensor) > mask_rate)
                masked_features = features_tensor * mask
                return masked_features
            except Exception as e:
                print(f"Warning: Failed to mask features: {e}")
                return features
    
    @staticmethod
    def mix_up_features(features, alpha=0.2):
        """
        对特征进行mixup增强
        
        参数:
        features: 形状为 (N, D) 的特征张量，N为节点数，D为特征维度
        alpha: Beta分布的参数
        
        返回:
        mixed_features: mixup后的特征
        """
        if features is None or not isinstance(features, th.Tensor):
            return features
            
        # 获取特征的设备
        device = features.device
            
        # 随机排列节点顺序
        indices = th.randperm(features.size(0), device=device)
        shuffled_features = features[indices]
        
        # 从Beta分布采样混合系数
        lam = np.random.beta(alpha, alpha)
        
        # 混合特征
        mixed_features = lam * features + (1 - lam) * shuffled_features
        return mixed_features


# 为现有的knn_graph函数添加增强版本
def augmented_knn_graph(disMat, k, dropout_rate=0.1, add_noise=False, noise_scale=0.1):
    """
    构建增强版k近邻图(带边丢弃和噪声)
    
    参数:
    disMat: 距离/相似度矩阵
    k: k近邻数
    dropout_rate: 随机丢弃的边的比例
    add_noise: 是否添加噪声
    noise_scale: 噪声标准差
    
    返回:
    adj: 增强后的邻接矩阵(稀疏格式)
    """
    from utils import knn_graph
    
    # 首先构建原始kNN图
    adj = knn_graph(disMat, k)
    
    # 添加噪声（如果需要）
    if add_noise:
        # 获取非零元素
        rows, cols = adj.nonzero()
        values = adj.data
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_scale, len(values))
        values = values + noise
        
        # 确保值在合理范围内
        values = np.clip(values, 0.01, 1.0)
        
        # 重建稀疏矩阵
        adj = sp.coo_matrix((values, (rows, cols)), shape=adj.shape)
    
    # 边丢弃
    if dropout_rate > 0:
        # 获取非零元素
        rows, cols = adj.nonzero()
        values = adj.data
        
        # 随机选择要保留的边
        num_edges = len(values)
        num_keep = max(1, int(num_edges * (1 - dropout_rate)))
        keep_indices = np.random.choice(num_edges, num_keep, replace=False)
        
        # 获取要保留的边
        rows_keep = rows[keep_indices]
        cols_keep = cols[keep_indices]
        values_keep = values[keep_indices]
        
        # 重建稀疏矩阵
        adj = sp.coo_matrix((values_keep, (rows_keep, cols_keep)), shape=adj.shape)
    
    # 确保对称性和自环
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    
    return adj

# 用于训练循环中的数据增强
def augment_graph_data(graph_data, aug_methods=None, aug_params=None):
    """
    对图数据进行增强
    
    参数:
    graph_data: 包含各种图和特征的字典
    aug_methods: 要应用的增强方法列表
    aug_params: 增强方法参数字典
    
    返回:
    augmented_data: 增强后的图数据
    """
    if aug_methods is None:
        aug_methods = ['edge_dropout']
    if aug_params is None:
        aug_params = {'edge_dropout_rate': 0.1}
    
    # 创建一个副本，避免修改原始数据
    augmented_data = {}
    for key, value in graph_data.items():
        if value is not None:
            # 深拷贝 tensor 和 graph 对象
            if isinstance(value, th.Tensor):
                augmented_data[key] = value.clone()
            elif isinstance(value, dgl.DGLGraph):
                augmented_data[key] = value.clone()
            else:
                augmented_data[key] = value
        else:
            augmented_data[key] = None
            
    # 应用指定的增强方法
    for method in aug_methods:
        if method == 'edge_dropout':
            # 对各种图应用边丢弃
            edge_dropout_rate = aug_params.get('edge_dropout_rate', 0.1)
            
            # 检查并处理每种图
            for graph_key in ['enc_graph', 'dec_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], dgl.DGLGraph):
                        try:
                            augmented_data[graph_key] = GraphAugmentation.random_edge_dropout(
                                augmented_data[graph_key], edge_dropout_rate)
                        except Exception as e:
                            print(f"Warning: Error applying edge_dropout to {graph_key}: {e}")
            
            # 处理稀疏张量图
            for graph_key in ['drug_graph', 'disease_graph', 
                             'drug_feature_graph', 'disease_feature_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], th.Tensor) and augmented_data[graph_key].is_sparse:
                        try:
                            augmented_data[graph_key] = GraphAugmentation.random_edge_dropout_sparse(
                                augmented_data[graph_key], edge_dropout_rate)
                        except Exception as e:
                            print(f"Warning: Error applying edge_dropout to sparse tensor {graph_key}: {e}")
        
        elif method == 'add_random_edges':
            # 在图中随机添加边
            add_edge_rate = aug_params.get('add_edge_rate', 0.05)
            
            for graph_key in ['enc_graph', 'dec_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    if isinstance(augmented_data[graph_key], dgl.DGLGraph):
                        try:
                            augmented_data[graph_key] = GraphAugmentation.add_random_edges(
                                augmented_data[graph_key], add_edge_rate)
                        except Exception as e:
                            print(f"Warning: Error applying add_random_edges to {graph_key}: {e}")
        
        elif method == 'feature_noise':
            # 向节点特征添加噪声
            feature_noise_scale = aug_params.get('feature_noise_scale', 0.1)
            sim_noise_scale = aug_params.get('sim_noise_scale', 0.05)
            
            for feat_key, noise_scale in [
                ('drug_feat', feature_noise_scale),
                ('disease_feat', feature_noise_scale),
                ('drug_sim_feat', sim_noise_scale),
                ('disease_sim_feat', sim_noise_scale)
            ]:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.feature_noise(
                            augmented_data[feat_key], noise_scale)
                    except Exception as e:
                        print(f"Warning: Error applying feature_noise to {feat_key}: {e}")
        
        elif method == 'graph_noise':
            # 向图结构添加噪声
            graph_noise_scale = aug_params.get('graph_noise_scale', 0.05)
            
            for graph_key in ['drug_graph', 'disease_graph', 
                             'drug_feature_graph', 'disease_feature_graph']:
                if graph_key in augmented_data and augmented_data[graph_key] is not None:
                    # 只处理稀疏张量
                    if isinstance(augmented_data[graph_key], th.Tensor) and augmented_data[graph_key].is_sparse:
                        try:
                            augmented_data[graph_key] = GraphAugmentation.sparse_graph_noise(
                                augmented_data[graph_key], graph_noise_scale)
                        except Exception as e:
                            print(f"Warning: Error applying graph_noise to {graph_key}: {e}")
        
        elif method == 'feature_masking':
            # 随机遮蔽特征
            feature_mask_rate = aug_params.get('feature_mask_rate', 0.1)
            
            for feat_key in ['drug_feat', 'disease_feat']:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.feature_masking(
                            augmented_data[feat_key], feature_mask_rate)
                    except Exception as e:
                        print(f"Warning: Error applying feature_masking to {feat_key}: {e}")
        
        elif method == 'mix_up':
            # 对节点特征应用mixup
            mixup_alpha = aug_params.get('mixup_alpha', 0.2)
            
            for feat_key in ['drug_feat', 'disease_feat']:
                if feat_key in augmented_data and augmented_data[feat_key] is not None:
                    try:
                        augmented_data[feat_key] = GraphAugmentation.mix_up_features(
                            augmented_data[feat_key], mixup_alpha)
                    except Exception as e:
                        print(f"Warning: Error applying mix_up to {feat_key}: {e}")
    
    return augmented_data