import torch as th
from sklearn import metrics

def evaluate(args, model, graph_data,
             drug_graph, drug_feat, drug_sim_feat,
             dis_graph, dis_feat, dis_sim_feat,
             drug_feature_graph=None, disease_feature_graph=None,
             return_predictions=False):
    """
    Evaluate model performance on given data, ensuring correct graph structures are used for evaluation.
    
    Args:
      args: Parameter object containing device and other information
      model: Model to evaluate
      graph_data: Dictionary containing graph data, format: {'test': [enc_graph, dec_graph, rating_values]}
      drug_graph: Drug similarity graph
      drug_feat: Drug features (embeddings)
      drug_sim_feat: Drug similarity features
      dis_graph: Disease similarity graph
      dis_feat: Disease features (embeddings)
      dis_sim_feat: Disease similarity features
      drug_feature_graph (optional): Drug feature kNN graph
      disease_feature_graph (optional): Disease feature kNN graph
      return_predictions (optional): Whether to return predictions, default False
      
    Returns:
      auc: Area under ROC curve
      aupr: Area under PR curve
      (optional) predictions: Tuple of predictions and true values (y_score, y_true)
    """
    # Get test data
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)
    
    # Ensure all graphs are moved to specified device
    drug_graph = drug_graph.to(args.device)
    dis_graph = dis_graph.to(args.device)
    if drug_feature_graph is not None:
        drug_feature_graph = drug_feature_graph.to(args.device)
    if disease_feature_graph is not None:
        disease_feature_graph = disease_feature_graph.to(args.device)
        
    # Set model to evaluation mode
    model.eval()
    with th.no_grad():
        # Forward pass
        pred_ratings, _, _, _, _ = model(
            enc_graph, dec_graph,
            drug_graph, drug_sim_feat, drug_feat,
            dis_graph, dis_sim_feat, dis_feat,
            drug_feature_graph, disease_feature_graph
        )
    
    # Convert predictions and true values to numpy arrays for evaluation
    y_score = pred_ratings.view(-1).cpu().numpy()
    y_true = rating_values.cpu().numpy()
    
    # Calculate ROC curve and AUROC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    
    # Calculate PR curve and AUPR
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    
    # Calculate other metrics that might be needed (optional but not returned)
    y_pred = (y_score >= 0.5).astype(int)
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    
    if return_predictions:
        return auc, aupr, (y_score, y_true)
    else:
        return auc, aupr