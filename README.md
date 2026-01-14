# CNN vs Vision Transformer (ViT) on CIFAR-10 — A Controlled Comparison (PyTorch)

This repository contains an end-to-end comparison between a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) on the CIFAR-10 dataset. Both models are trained and evaluated under the same dataset split and comparable evaluation metrics to study their learning behavior, generalization, and error patterns.

## Dataset
- CIFAR-10: 60,000 color images (32×32), 10 classes
- Train: 50,000 images
- Test: 10,000 images

## Project Goals
1. Build a strong CNN baseline and improve it through architectural iteration (e.g., BatchNorm, depth).
2. Implement a ViT model on the same dataset.
3. Compare CNN vs ViT using consistent evaluation:
   - overall test accuracy
   - confusion matrix
   - per-class accuracy
   - precision/recall/F1 report
   - qualitative inference examples

## Repository Structure
- `cnn/` : CNN training, inference, evaluation, and architectures
- `vit/` : ViT training, inference, evaluation, and architectures
- `common/` : shared dataset + metrics + utilities
- `results/` : saved plots and evaluation outputs for both models

## CNN Summary (Completed)
- Final CNN architecture: 4 × (Conv → BatchNorm → ReLU → Pool) + dense classifier
- Achieved ~80% test accuracy on CIFAR-10
- Includes inference visualization and detailed evaluation outputs

## ViT Summary (In Progress / Planned)
- ViT implementation on CIFAR-10 using patch embeddings + transformer encoder
- Will report comparable metrics and analyze differences in error patterns and data efficiency

## How to Install
```bash
pip install -r requirements.txt
