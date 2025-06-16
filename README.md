# AdaDR: Adaptive Drug-Disease Association Prediction

[![DGL](https://img.shields.io/badge/DGL-0.8+-green.svg)](https://www.dgl.ai/)

A deep learning framework for drug-disease association prediction using graph neural networks with data augmentation and multiple similarity graphs.

## 🔬 Overview

AdaDR predicts potential drug-disease associations by leveraging:
- **Multi-Graph Learning**: Drug similarity, disease similarity, and feature-based graphs
- **Data Augmentation**: Edge dropout, feature noise, graph perturbations
- **Cross-Validation**: 10-fold CV for robust evaluation
- **Novel Prediction**: Identifies top-k novel drug-disease pairs

## 📋 Requirements

```bash
pip install torch>=1.8.0
pip install dgl>=0.8.0
pip install numpy pandas scikit-learn scipy
```

## 🚀 Quick Start

### Basic Training
```bash
python main_train.py --data_name lrssl --device 0
```

### Ablation Study
```bash
python ablation_study.py
```

## 📁 Project Structure

```
├── main_train.py          # Main training script
├── ablation_study.py      # Comprehensive ablation experiments
├── data_loader.py         # Data loading and preprocessing
├── model.py              # Neural network architectures
├── layers.py             # Custom GNN layers
├── evaluation.py         # Model evaluation metrics
├── augmentation.py       # Data augmentation techniques
├── utils.py              # Utility functions
├── run_experiments.sh    # Batch experiment runner
└── README.md            # This file
```

## 🎯 Datasets

Place your datasets in `./raw_data/drug_data/`:
- **Gdataset**: `Gdataset/Gdataset.mat`
- **Cdataset**: `Cdataset/Cdataset.mat`  
- **LRSSL**: `lrssl/lrssl.mat`

Expected `.mat` file structure:
```matlab
didr          % Drug-disease association matrix
drug          % Drug similarity matrix
disease       % Disease similarity matrix
drug_embed    % Pre-trained drug embeddings (optional)
disease_embed % Pre-trained disease embeddings (optional)
Wrname        % Drug names (optional)
```

## 📈 Output

### Training Results
- `fold_metrics.csv`: Per-fold AUROC/AUPR scores
- `best_model_fold{X}.pth`: Best model for each fold
- `test_metric{X}.csv`: Training progress logs

### Novel Predictions
- `top{K}_novel_predictions_fold{X}.csv`: Top-K predictions per fold
- `combined_top_predictions.csv`: Aggregated predictions across folds


**Key Components:**
- **GCMC Layers**: Handle drug-disease bipartite graphs
- **Feature GCNs**: Process similarity-based graphs
- **Attention Fusion**: Combines multiple representations
- **MLP Decoder**: Predicts association probabilities



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

