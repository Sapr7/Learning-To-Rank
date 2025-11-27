# Learning-To-Rank with Transformer Models

This repository contains an implementation of Learning-to-Rank methods based on the Transformer Encoder architecture. The project is designed for ranking documents by relevance to queries using deep neural networks.

## 📋 Project Description

The project implements approaches to the ranking task (Learning-to-Rank) using transformer architectures. The model is based on the Encoder architecture and supports various loss functions (pointwise, listwise, combined), enabling efficient training of models for document ranking.

### Key Features

- **Transformer Encoder model** for document ranking
- **Multiple loss functions**: Pointwise (Cross-Entropy), Listwise (ListNet), Combined Loss
- **Comprehensive metric evaluation**: NDCG@5, NDCG@10, NDCG (full), Recall@5, Recall@10, Recall (full), MRR
- **Analysis utilities**: inference time measurement, memory usage estimation
- **Fine-tuning support** for models
- **Visualization** of training results and comparison of different architectures

## 🏗️ Model Architecture

The model is a Transformer-based Encoder:

```
Input (num_docs × num_features)
    ↓
Input Projection (Linear)
    ↓
Transformer Blocks (× N layers)
    ├─ Multi-Head Self-Attention
    ├─ Residual Connection + Layer Norm
    ├─ Feed-Forward Network
    └─ Residual Connection + Layer Norm
    ↓
Output Layer (Linear → num_classes)
    ↓
Scores (num_docs × num_classes)
```

### Model Parameters

- `d_model`: Model dimension (default: 512)
- `n_heads`: Number of attention heads (default: 2-4)
- `n_layers`: Number of transformer blocks (default: 2)
- `ffn_hidden`: FFN hidden layer dimension (default: 512)
- `input_dim`: Input feature dimension (depends on dataset)
- `output_dim`: Number of relevance classes (default: 5)
- `dropout_rate`: Dropout coefficient (default: 0.15)

### Installing Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
pip install thop  # For FLOPs counting (optional)
```

## Usage

### 1. Data Preprocessing

Data should be in pickle file format with the following structure:
- `fl_features`: document features
- `labels`: relevance labels (0-4)
- `query_id`: query identifiers

Usage example:

```python
from utils.preprocess import preprocess_data

train_data = preprocess_data(
    file_path='path/to/train.pkl',
    num_docs=140,        # Maximum number of documents per query
    which=0,             # Dataset index (0 for train, -1 for test)
    is_shuffle=True,     # Whether to shuffle documents
    device='cuda'
)
```

### 2. Creating the Model

```python
from utils.Encoder_model import make_Encoder_model

model = make_Encoder_model(
    d_model=512,
    n_heads=2,
    n_layers=2,
    ffn_hidden=512,
    input_dim=699,       # Feature dimension
    output_dim=5,        # Number of classes
    dropout_rate=0.15,
    device='cuda'
)
```

### 3. Training the Model

The main training pipeline is in `Training and evaluation.ipynb`:

```python
from utils.train_eval_utils import train_eval
from utils.loss_mask_utils import Combined_Loss, create_mask
from sklearn.metrics import ndcg_score

loss_fn = Combined_Loss(theta=0.01, num_of_labels=5, distribution='polynomial', degree=2)

train_params = {
    'train_loader': train_loader,
    'model': model,
    'optimizer': optimizer,
    'loss_fn': loss_fn,
    'num_epochs': 25,
    'create_mask': create_mask,
    'val_loader': val_loader,
    'score_fn': ndcg_score,
    'name': 'best_model'
}

losses, metrics = train_eval(**train_params)
```

### 4. Metric Evaluation

The `train_eval` function automatically computes multiple ranking metrics:

#### Ranking Quality Metrics

- **NDCG@5**: Normalized Discounted Cumulative Gain on top-5 documents
- **NDCG@10**: NDCG on top-10 documents
- **NDCG (full)**: NDCG on all documents in the ranking

#### Recall Metrics

- **Recall@5**: Proportion of relevant documents found in top-5 results
- **Recall@10**: Proportion of relevant documents found in top-10 results
- **Recall (full)**: Proportion of relevant documents found in the entire ranking

#### Rank-based Metrics

- **MRR (Mean Reciprocal Rank)**: Average of the reciprocal ranks of the first relevant document for each query

All metrics are computed during validation and displayed in the console output. The metrics dictionary returned by `train_eval` contains lists of all metric values for each epoch, enabling detailed analysis of model performance over time.

### 5. Fine-tuning

To fine-tune an existing model, use `finetune.ipynb`:

```python
from utils.preprocess import Dataset_for_finetune, preprocess_for_finetune
from utils.train_eval_utils import train_eval
from utils.loss_mask_utils import cross_entropy_for_finetune

# Load model
model = make_Encoder_model(**model_params)
state_dict = torch.load('path/to/model.pth')
model.load_state_dict(state_dict)

# Fine-tuning with new loss function
loss_fn = cross_entropy_for_finetune
# ... further configuration
```

## 📊 Loss Functions

The project supports several loss functions:

### 1. Pointwise Loss (Cross-Entropy)
A classical approach treating ranking as classification by relevance classes.

```python
from utils.loss_mask_utils import Cross_Entropy_point

loss_fn = Cross_Entropy_point(num_of_label=5)
```

### 2. Listwise Loss (ListNet)
A listwise approach that considers the relevance distribution in the list.

```python
from utils.loss_mask_utils import ListNet_Loss

loss_fn = ListNet_Loss(distribution='polynomial', degree=2)
# or
loss_fn = ListNet_Loss(distribution='softmax')
```

### 3. Combined Loss
A combination of pointwise and listwise losses.

```python
from utils.loss_mask_utils import Combined_Loss

loss_fn = Combined_Loss(
    theta=0.01,                    # Weight for listwise loss
    num_of_labels=5,
    distribution='polynomial',
    degree=2
)
```

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation metrics for ranking tasks:

### Available Metrics

1. **NDCG (Normalized Discounted Cumulative Gain)**
   - Measures ranking quality considering position and relevance
   - Computed at different cutoffs: @5, @10, and full ranking

2. **Recall@k**
   - Measures the proportion of relevant documents retrieved in top-k results
   - Useful for understanding coverage of relevant items
   - Configurable relevance threshold (default: > 0.0)

3. **MRR (Mean Reciprocal Rank)**
   - Measures the average reciprocal rank of the first relevant document
   - Particularly useful when the position of the first relevant result matters
   - Returns 0 if no relevant documents are found

### Customizing Metrics

You can customize the relevance threshold for Recall and MRR:

```python
from utils.train_eval_utils import evaluate

# Evaluate with custom relevance threshold
avg_ndcg5, avg_ndcg10, avg_ndcg, avg_recall5, avg_recall10, avg_recall, avg_mrr = evaluate(
    val_loader, 
    model, 
    ndcg_score, 
    create_mask,
    relevance_threshold=0.5  # Documents with score > 0.5 are considered relevant
)
```

## Performance Analysis

### Inference Time Measurement

Use `inference_time.ipynb` for model performance analysis:

- Static memory estimation (parameters, buffers)
- Peak memory usage during inference
- Forward pass execution time

### Memory Measurement

```python
from inference_time import estimate_inference_memory_static, measure_inference_peak_memory

# Static estimation
static_info = estimate_inference_memory_static(model)
print(static_info['pretty'])

# Peak usage during inference
memory_info = measure_inference_peak_memory(model, sample_input, warmup=5, steps=10)
print(memory_info['cuda_peak']['pretty'])
```

## Experiment Results

The `done pictures/` folder contains results from experiments with various:
- Model architectures
- Loss functions (pointwise, listwise, combined)
- Hyperparameters (dropout, polynomial degree)
- Datasets (Web10k, Istella)

##  Utilities

### Creating Padding Mask

```python
from utils.loss_mask_utils import create_mask

mask = create_mask(input_tensor)  # Boolean mask for documents
```

### Computing Metrics Manually

You can compute metrics individually using utility functions:

```python
from utils.train_eval_utils import compute_recall_at_k, compute_mrr
import numpy as np

# Example: Compute Recall@10
y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.0])  # Ground truth relevance
y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])  # Predicted scores

recall_10 = compute_recall_at_k(y_true, y_pred, k=10, relevance_threshold=0.0)
print(f"Recall@10: {recall_10:.4f}")

# Example: Compute MRR
mrr = compute_mrr(y_true, y_pred, relevance_threshold=0.0)
print(f"MRR: {mrr:.4f}")
```

### Data Preprocessing

The `Dataset_for_transformer` class creates a Dataset for PyTorch DataLoader:

```python
from utils.preprocess import Dataset_for_transformer
from torch.utils.data import DataLoader

dataset = Dataset_for_transformer(preprocessed_data)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
```