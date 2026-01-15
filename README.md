# CNN vs Vision Transformer (ViT) on CIFAR-10 (PyTorch)

This repository presents a structured computer vision project focused on building, analyzing, and comparing two fundamentally different image classification paradigms: Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). The CNN component has been fully implemented and evaluated, while the ViT component is planned as the second phase of the project, enabling a principled and fair comparison on the same dataset.

---

## Dataset
- **CIFAR-10**: 60,000 color images (32×32 pixels) across 10 classes  
- Training set: 50,000 images  
- Test set: 10,000 images  

The dataset is downloaded locally during execution and is intentionally excluded from version control.

---

## Project Objectives
1. Build a CNN from scratch and iteratively improve it through architectural refinement.
2. Rigorously evaluate CNN performance using quantitative and qualitative metrics.
3. Implement a Vision Transformer (ViT) on the same dataset.
4. Compare CNN and ViT in terms of accuracy, error patterns, and inductive bias.

---

## CNN Component (Completed)

### Architecture
The final CNN architecture consists of:
- 4 × (Convolution → Batch Normalization → ReLU → Max Pooling)
- Fully connected classifier with dropout
- Designed specifically for small-resolution images (32×32)

### Training Details
- Framework: PyTorch
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Hardware: CUDA-enabled NVIDIA GPU
- Training performed locally (not on cloud platforms)

### Performance
- Final test accuracy: **~80%**
- Significant improvement over the baseline CNN through increased depth and batch normalization

### Evaluation & Analysis
The CNN model is evaluated using:
- Overall test accuracy
- Confusion matrix
- Per-class accuracy
- Precision, recall, and F1-score
- Qualitative inference visualization on unseen test images
- Visualization of learned convolutional filters (before and after training)

## How to Run the CNN Project (Local)

This project was trained and evaluated locally (not on Google Colab).

### 1. Install dependencies
```bash
pip install -r requirements.txt

---
```
## Vision Transformer (ViT) Component (Planned)
The second phase of this project will implement a Vision Transformer on CIFAR-10 using patch embeddings and transformer encoder blocks. The goal is to compare ViT performance against the CNN baseline under identical data and evaluation conditions, focusing on differences in learning behavior, data efficiency, and error characteristics.

## Repository Structure
