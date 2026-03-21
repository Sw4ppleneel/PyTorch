# PyTorch Deep Learning Journey

A comprehensive learning repository documenting the progression from PyTorch fundamentals to production-level deep learning models. This repo contains hands-on implementations of neural networks, computer vision models, transfer learning, and experiment tracking.

## 📚 Repository Structure

```
PyTorch/
├── PyTorchFol/                 # Main working directory
│   ├── Notebooks/
│   │   ├── Classification.ipynb          # Binary & multi-class classification
│   │   ├── PyT.ipynb                     # PyTorch fundamentals
│   │   ├── ComputerVision.ipynb          # CNN architectures (TinyVGG)
│   │   ├── Custom_Dataset.ipynb          # Custom Dataset classes
│   │   ├── Modular.ipynb                 # Modular project structure
│   │   ├── Transfer_learning.ipynb       # Transfer learning with EfficientNet
│   │   └── Experiment_tracking.ipynb     # TensorBoard experiment tracking
│   ├── going_modular/                    # Production-ready modules
│   │   ├── data_setup.py                 # DataLoader creation
│   │   ├── engine.py                     # Training/testing loops
│   │   ├── model_builder.py              # Model architectures
│   │   ├── predictions.py                # Inference utilities
│   │   ├── train.py                      # Main training script
│   │   └── utils.py                      # Helper functions
│   ├── data/                             # Datasets
│   │   ├── pizza_steak_sushi/            # Full dataset (3 classes)
│   │   ├── pizza_steak_sushi_20_percent/ # 20% subset
│   │   └── 04-pizza-dad.jpeg             # Custom test image
│   ├── Models/                           # Saved model checkpoints
│   ├── FashionMNIST/                     # Fashion-MNIST dataset
│   ├── helper_functions.py               # Visualization & utilities
│   └── runs/                             # TensorBoard logs
```

## 🚀 Key Projects

### 1. **Classification Fundamentals** (`Classification.ipynb`)
- Binary classification on synthetic circle data
- Multi-class classification on blob datasets
- Loss functions: BCEWithLogitsLoss, CrossEntropyLoss
- Optimizers comparison: SGD vs Adam

**Results:** 100% accuracy on synthetic datasets

### 2. **Computer Vision** (`ComputerVision.ipynb`)
- **TinyVGG**: Custom CNN architecture from scratch
- Image preprocessing & augmentation
- Training on pizza/steak/sushi dataset (75 images, 3 classes)
- Device management (CPU/MPS)

**Results:** ~87% test accuracy with small dataset

### 3. **Transfer Learning** (`Transfer_learning.ipynb`)
**Model Comparison (Same Seed: 1000):**

| Model | Architecture | Accuracy | Time | Notes |
|-------|--------------|----------|------|-------|
| v0 | EfficientNet-B0 | 87.69% | 15.67s | Faster, lightweight |
| v1 | EfficientNet-B2 | 90.72% | 21.09s | More parameters, better accuracy |

**Key Features:**
- Frozen backbone (ImageNet pretrained)
- Custom classifier head (2 layers)
- Proper device handling (MPS/CPU)
- Model save/load with state dict
- Inference pipeline with visualization

### 4. **Experiment Tracking** (`Experiment_tracking.ipynb`)
- **TensorBoard Integration**: Real-time metrics visualization
- **Systematic Experiments**: 8 model configurations
- **Variables Tested:**
  - Models: EfficientNet-B0, EfficientNet-B2
  - Data: 10%, 20% training splits
  - Epochs: 5, 10
  
**Experiment Results Saved to:** `runs/` directory with timestamped organization

### 5. **Modular Architecture** (`going_modular/`)
Production-ready code organization:
- **data_setup.py**: ImageFolder DataLoaders with transforms
- **engine.py**: Device-agnostic training/testing loops
- **model_builder.py**: Reusable model creation functions
- **predictions.py**: Inference utilities
- **train.py**: Main training script

## 🛠️ Technologies & Libraries

```python
# Core
PyTorch >= 1.9.0
torchvision >= 0.10.0
NumPy, Pandas

# Visualization
Matplotlib, Seaborn
TensorBoard

# Tools
torchinfo       # Model summaries
Pillow          # Image processing
requests        # Data downloads
```

## 📊 Dataset: Pizza, Steak, Sushi

**Source:** Custom dataset from mrdbourke/pytorch-deep-learning

**Details:**
- **Classes:** 3 (Pizza, Steak, Sushi)
- **Train Images:** ~67 (full), ~13-14 (10%), ~27-28 (20%)
- **Test Images:** ~33
- **Image Size:** Variable (resized to 224x224 for models)
- **Splits:** 80% train, 20% test

## 🔧 How to Use

### Setup Environment
```bash
cd PyTorch/PyTorchFol
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Notebooks
```bash
jupyter notebook Transfer_learning.ipynb
jupyter notebook Experiment_tracking.ipynb
```

### Use Modular Code
```python
from going_modular import data_setup, engine, model_builder
from pathlib import Path
import torch

# Setup
device = torch.device("mps" if torch.mps.is_available() else "cpu")
train_dir = Path("data/pizza_steak_sushi/train")
test_dir = Path("data/pizza_steak_sushi/test")

# Create DataLoaders
train_dl, test_dl, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, 
    test_dir=test_dir,
    batch_size=32
)

# Create model
model = model_builder.create_effnetb2()

# Train
results = engine.train(
    model=model,
    train_dataloader=train_dl,
    test_dataloader=test_dl,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    loss_fn=torch.nn.CrossEntropyLoss(),
    epochs=10,
    device=device
)
```

### Load Trained Model
```python
import torch
import torchvision
from pathlib import Path

device = torch.device("mps" if torch.mps.is_available() else "cpu")

# Load v0 model (EfficientNet-B0)
model_v0 = torchvision.models.efficientnet_b0(weights=None)
model_v0.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(in_features=1280, out_features=3)
)
model_v0.load_state_dict(torch.load(
    "Models/transfer_trained_model_v0", 
    map_location=device
))
model_v0.to(device).eval()
```

### View TensorBoard Logs
```bash
cd PyTorchFol
tensorboard --logdir runs
# Visit: http://localhost:6006
```

## 💡 Key Learnings

### Device Management
- ✅ Always use device objects, not strings
- ✅ Move model AND data to device
- ✅ Use `map_location` when loading checkpoints
- ✅ Set device in all train/test/inference functions

### Transfer Learning Best Practices
- ✅ Freeze backbone for small datasets (<100 images)
- ✅ Only train custom classifier head
- ✅ Use same transforms as pretraining (ImageNet normalization)
- ✅ Set `num_workers=0` for small datasets on macOS

### Model Serialization
```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load with device safety
model.load_state_dict(torch.load("model.pth", map_location=device))
```

### Training Patterns
- Train/eval modes: `model.train()` / `model.eval()`
- Inference context: `torch.inference_mode()` / `torch.no_grad()`
- Seeding: Set both `torch.manual_seed()` and `torch.mps.manual_seed()`

## 📈 Experiment Results Summary

**Best Model:** EfficientNet-B2 on 20% data, 10 epochs
- **Test Accuracy:** 90.72%
- **Training Time:** 21.09 seconds
- **Checkpoint:** `Models/transfer_trained_model_v1`

**Trade-offs:**
- **Speed vs Accuracy:** B0 is 5s faster, B2 is 3% more accurate
- **Data Size:** 20% data achieves similar results to 10% with more epochs
- **Efficiency:** Both models <50MB for production deployment

## 🐛 Common Issues & Solutions

### Issue: "Input type MPS doesn't match weight type CPU"
**Solution:** Ensure model and data on same device
```python
model = model.to(device)
data = data.to(device)
```

### Issue: DataLoader workers crash on macOS
**Solution:** Set `num_workers=0`
```python
DataLoader(dataset, batch_size=32, num_workers=0)
```

### Issue: Type hint errors with return types
**Solution:** Use class reference, not instantiation
```python
# ❌ Wrong
def func() -> SummaryWriter():

# ✅ Correct
def func() -> SummaryWriter:
```

## 🔄 Model Checkpoints

| File | Architecture | Size | Accuracy | Epochs |
|------|--------------|------|----------|--------|
| transfer_trained_model_v0 | EfficientNet-B0 | ~22MB | 87.69% | 10 |
| transfer_trained_model_v1 | EfficientNet-B2 | ~35MB | 90.72% | 10 |
| 07_effnetb2_data_20_percent_10_epochs.pth | EfficientNet-B2 | ~35MB | 90.72% | 10 |

## 📝 Notebooks Overview

| Notebook | Topic | Status | Key Output |
|----------|-------|--------|------------|
| PyT.ipynb | Fundamentals | ✅ Complete | Tensor operations |
| Classification.ipynb | Binary/Multi-class | ✅ Complete | 100% synthetic accuracy |
| Custom_Dataset.ipynb | Dataset creation | ✅ Complete | Custom Dataset class |
| ComputerVision.ipynb | TinyVGG CNN | ✅ Complete | 87% pizza/steak/sushi |
| Transfer_learning.ipynb | EfficientNet transfer | ✅ Complete | v0 & v1 models, inference |
| Experiment_tracking.ipynb | Systematic experiments | ✅ Complete | 8 experiments, TensorBoard logs |
| Modular.ipynb | Code organization | ✅ Complete | Production structure |

## 🎯 Future Improvements

- [ ] Data augmentation strategies (Mixup, Cutout)
- [ ] Hyperparameter optimization (Ray Tune)
- [ ] Cross-validation for small datasets
- [ ] Model ensemble techniques
- [ ] Grad-CAM visualization
- [ ] Mobile deployment (ONNX export)
- [ ] API wrapper for inference

## 📚 References

- [PyTorch Official Docs](https://pytorch.org/docs/)
- [Learn PyTorch Course](https://www.learnpytorch.io/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 👤 Author Notes

This repository documents a learning journey from fundamental PyTorch concepts to production-level deep learning. Each notebook builds on previous knowledge, progressing from simple classification → custom architectures → transfer learning → systematic experimentation.

**Key Achievements:**
- ✅ Device-agnostic code (CPU/MPS)
- ✅ Production-ready modular structure
- ✅ Systematic experiment tracking
- ✅ Model comparison with fair conditions
- ✅ Proper save/load checkpointing

## 📄 License

Educational/Personal Use

---

**Last Updated:** March 2026  
**Python Version:** 3.9+  
**PyTorch Version:** 1.9+
