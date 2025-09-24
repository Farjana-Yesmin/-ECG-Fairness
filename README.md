# -ECG-Fairness
Fairness-Aware ECG-Based Disease Prediction in Wearable Systems

This repository contains the official implementation of the paper "Fairness-Aware Representation Learning for ECG-Based Disease Prediction in Wearable Systems".

📖 Paper Abstract

Machine learning models for electrocardiogram (ECG)-based disease prediction, increasingly integrated into wearable health devices, often exhibit demographic biases, exacerbating healthcare disparities. We propose a fairness-aware representation learning framework using adversarial debiasing tailored for biosignals, focusing on inferior myocardial infarction (IMI) classification. Our approach addresses biosignal-specific challenges, such as physiological variations across sex and age, differing from imaging or tabular data.

🚀 Quick Start

Prerequisites

Python 3.8+
PyTorch 2.6.0+
WFDB package for ECG data handling
Google Colab (for free-tier GPU access)
Installation


# Install required packages
pip install wfdb torch==2.6.0 pandas scikit-learn matplotlib

# For GPU support (if available)
pip install torch==2.6.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
Dataset Setup

The code uses the PTB-XL ECG dataset. The dataset will be automatically downloaded and preprocessed:


# Dataset will be downloaded from PhysioNet
# 20% subsample (4,367 records) used due to computational constraints
Running the Code

Execute the Jupyter notebook EAI_HealthWear_2025.ipynb sequentially:

Installation & Setup: Install dependencies and configure environment
Data Preparation: Download and preprocess PTB-XL dataset
Model Training: Train fairness-aware ECG classification model
Evaluation: Benchmark against state-of-the-art methods
Visualization: Generate results and fairness metrics
🏗️ Model Architecture

Our framework consists of three main components:

Encoder: 1D CNN (12→64→128 channels) + Fully Connected layer
Classifier: Binary classifier for IMI detection
Adversarial Networks: Demographic attribute predictors with gradient reversal

Input ECG → Encoder → [Classifier, Adversary_Sex, Adversary_Age]
📊 Results

Performance Metrics (20% PTB-XL Subsample)

Method	AUROC	Accuracy	F1 Score	DI-Sex
Baseline CNN	0.92	0.85	0.55	0.23
Reweighting	0.90	0.84	0.53	0.30
FairMixup	0.88	0.83	0.52	0.35
AdvFair	0.85	0.82	0.51	0.45
GroupDRO	0.84	0.81	0.50	0.50
Ours	0.8472	0.81	0.50	0.71
Per-Group Performance

Demographic Group	AUROC	F1 Score
Male	0.83	0.51
Female	0.81	0.49
Age <40	0.79	0.42
Age 40-59	0.84	0.52
Age ≥60	0.82	0.50
🎯 Key Features

Fairness-Aware Learning: Adversarial debiasing for demographic parity
ECG-Specific Architecture: Tailored for biosignal characteristics
Wearable-Optimized: Lightweight design suitable for edge devices
Comprehensive Benchmarking: Comparison with 5 state-of-the-art methods
Reproducible Setup: Complete code and configuration for easy replication
📁 Project Structure


├── EAI_HealthWear_2025.ipynb      # Main experiment notebook
├── requirements.txt               # Python dependencies
├── models/                        # Model architectures
│   ├── encoder.py                 # ECG signal encoder
│   ├── classifier.py              # Disease classifier
│   └── adversary.py               # Adversarial networks
├── utils/                         # Utility functions
│   ├── data_loader.py             # ECG data loading and preprocessing
│   ├── metrics.py                 # Fairness and performance metrics
│   └── visualization.py           # Plotting and result visualization
└── results/                       # Generated results and figures
⚙️ Configuration

Key hyperparameters (set in the notebook):


config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 20,
    'lambda_adversary': 0.3,  # Fairness-accuracy trade-off
    'signal_length': 1000,    # 10-second segments at 100Hz
    'num_leads': 12           # Standard 12-lead ECG
}
📈 Fairness Metrics

We evaluate using standard fairness metrics:

Demographic Parity Difference (DPD)
Equal Opportunity Difference (EOD)
Disparate Impact (DI)
🔬 Experimental Setup

Dataset: PTB-XL ECG dataset (20% subsample, 4,367 records)
Hardware: Google Colab free tier (T4 GPU, ~12GB RAM)
Framework: PyTorch 2.6.0
Validation: 5-fold cross-validation

📝 Citation

If you use this code in your research, please cite our paper:

bibtex
@inproceedings{healthwear2025,
  title={Fairness-Aware Representation Learning for ECG-Based Disease Prediction in Wearable Systems},
  author={Farjana Yesmin and Nusrat Shirmin},
  year={2025}
}

🤝 Contributing
We welcome contributions! Please feel free to submit issues and pull requests.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

PTB-XL dataset providers
Google Colab for computational resources
Contributors and reviewers

