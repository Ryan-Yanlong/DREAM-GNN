import torch as th
from sklearn import metrics

def evaluate(args, model, graph_data,
             drug_graph, drug_feat, drug_sim_feat,
             dis_graph, dis_feat, dis_sim_feat,
             drug_feature_graph=None, disease_feature_graph=None,
             return_predictions=False):
    """
    评估模型在给定数据上的性能，确保使用正确的图结构进行评估。
    
    参数：
      args: 包含设备(device)等信息的参数对象。
      model: 待评估模型。
      graph_data: 包含图数据的字典，格式为 {'test': [enc_graph, dec_graph, rating_values]}。
      drug_graph: 药物相似性图。
      drug_feat: 药物特征（embedding）。
      drug_sim_feat: 药物相似性特征。
      dis_graph: 疾病相似性图。
      dis_feat: 疾病特征（embedding）。
      dis_sim_feat: 疾病相似性特征。
      drug_feature_graph (可选): 药物特征 kNN 图。
      disease_feature_graph (可选): 疾病特征 kNN 图。
      return_predictions (可选): 是否返回预测值，默认为 False。
      
    返回：
      auc: ROC 曲线下面积。
      aupr: PR 曲线下面积。
      (可选) predictions: 预测值和真实值的元组 (y_score, y_true)。
    """
    # 获取测试数据
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)
    
    # 确保所有图均搬到指定设备上
    drug_graph = drug_graph.to(args.device)
    dis_graph = dis_graph.to(args.device)
    if drug_feature_graph is not None:
        drug_feature_graph = drug_feature_graph.to(args.device)
    if disease_feature_graph is not None:
        disease_feature_graph = disease_feature_graph.to(args.device)
        
    # 设置模型为评估模式
    model.eval()
    with th.no_grad():
        # 前向传播
        pred_ratings, _, _, _, _ = model(
            enc_graph, dec_graph,
            drug_graph, drug_sim_feat, drug_feat,
            dis_graph, dis_sim_feat, dis_feat,
            drug_feature_graph, disease_feature_graph
        )
    
    # 将预测值与真实值转换为 numpy 数组用于评估
    y_score = pred_ratings.view(-1).cpu().numpy()
    y_true = rating_values.cpu().numpy()
    
    # 计算 ROC 曲线和 AUROC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    
    # 计算 PR 曲线和 AUPR
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    
    # 计算其他可能需要的指标（可选但不返回）
    y_pred = (y_score >= 0.5).astype(int)
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    
    if return_predictions:
        return auc, aupr, (y_score, y_true)
    else:
        return auc, aupr