# loan-default-prediction-ml

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced loan default prediction system combining Deep Learning and Reinforcement Learning for optimal financial decision-making. Achieves 78% AUC with memory-optimized processing for large datasets.**

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Business Impact](#business-impact)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a comprehensive machine learning solution for loan default prediction using the LendingClub dataset. The system combines traditional supervised deep learning with reinforcement learning principles to create an intelligent loan approval system that maximizes financial returns while effectively managing risk.

**Key Achievements:**
- 🚀 **78% AUC Score** - Excellent discriminative capability
- 📈 **45% F1-Score** for defaulters - Significant improvement from baseline
- 💰 **68% Overall Accuracy** - Industry-competitive performance
- 🔧 **Memory-Optimized** - Handles datasets with 2M+ records efficiently
- 🧠 **Hybrid Decision System** - Combines risk assessment with profit optimization

## 🎯 Problem Statement

Financial institutions face the critical challenge of making loan approval decisions that balance profitability with risk management. This project addresses:

1. **Risk Assessment**: Predict loan default probability with high accuracy
2. **Profit Optimization**: Maximize expected returns through intelligent decision policies  
3. **Class Imbalance**: Handle the inherent imbalance in financial datasets (20% default rate)
4. **Scalability**: Process large-scale real-world datasets efficiently
5. **Business Integration**: Provide actionable insights for financial decision-making

## ✨ Features

### 🧠 Advanced Machine Learning
- **Enhanced Deep Learning**: MLP with BatchNorm, Dropout, and class weighting
- **Memory Optimization**: Chunked data processing for large datasets
- **Class Imbalance Solutions**: Weighted loss functions and optimal threshold selection
- **Feature Engineering**: Domain-specific financial features and encoding

### 📊 Comprehensive Analysis
- **Performance Evaluation**: ROC/AUC analysis and precision-recall curves
- **Policy Comparison**: Deep Learning vs Reinforcement Learning decision analysis
- **Business Metrics**: Financial return projections and risk assessments
- **Visualization Suite**: Training history and decision boundary plots

### 🚀 Production-Ready
- **Early Stopping**: Prevents overfitting with F1-score monitoring
- **Model Persistence**: Save and load trained models
- **Extensive Logging**: Detailed training and evaluation metrics
- **Error Handling**: Robust data processing with comprehensive error management

## 📊 Performance Metrics

### Deep Learning Model Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **AUC** | 0.78 | Excellent discriminative power |
| **F1-Score (Defaulters)** | 0.45 | Strong minority class detection |
| **Precision (Defaulters)** | 0.35 | Reasonable false positive rate |
| **Recall (Defaulters)** | 0.64 | Catches 64% of actual defaults |
| **Overall Accuracy** | 68% | Industry-competitive performance |

### Detailed Classification Report

          precision    recall  f1-score   support
     0.0       0.89      0.69      0.78     62468
     1.0       0.35      0.64      0.45     15765
accuracy                           0.68     78233


## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (recommended for full dataset)
- CUDA-compatible GPU (optional, for faster training)


