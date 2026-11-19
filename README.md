# Improving Cross-Dataset Generalization of Deepfake Detection Using Frequency-Domain Feature Learning

A Deep Learning Research Project on Deepfake Detection using **Spatial and Frequency-Domain Feature Learning**, with an emphasis on improving generalization across different datasets.

---

## 1. Project Overview

Deepfake technology has made it increasingly difficult to distinguish authentic media from artificially manipulated content. Although deep learning models have achieved promising results in detecting manipulated images and videos, their performance often decreases when evaluated on datasets different from those used during training.

This research project investigates a hybrid deepfake detection framework that combines spatial-domain and frequency-domain features to improve the detection of manipulated facial images and study cross-dataset generalization.

The proposed framework integrates an **Xception-based RGB feature extraction branch** with a **Frequency CNN branch**, followed by a fusion network that combines the extracted representations for final classification.

The experiments are conducted using two datasets:

- FaceForensics++ (FF++)
- WildDeepfake

The project evaluates both in-dataset performance and cross-dataset generalization to understand the model's robustness under different data distributions.

## 2. Research Objectives

The primary objectives of this research are:

- Develop a hybrid deepfake detection architecture combining spatial and frequency-domain information.
- Extract meaningful facial representations using a pretrained Xception network.
- Investigate frequency-domain features using FFT-based preprocessing and a dedicated CNN.
- Combine spatial and frequency features through a feature fusion network.
- Evaluate model performance on FaceForensics++ and WildDeepfake.
- Investigate the impact of dataset distribution differences on cross-dataset performance.
- Identify limitations and potential improvements for more generalizable deepfake detection.

## 3. Proposed Methodology

The proposed framework consists of three major components.

### 3.1 Spatial Feature Extraction – Xception

The Xception network is used to extract spatial-domain features from RGB facial images.

- Uses a pretrained Xception architecture.
- Learns spatial patterns and visual inconsistencies.
- Generates a 2048-dimensional feature representation.

### 3.2 Frequency Feature Extraction – Frequency CNN

The frequency branch focuses on identifying manipulation-related information in the frequency domain.

- Applies Fast Fourier Transform (FFT) preprocessing.
- Extracts frequency-domain representations.
- Uses a dedicated FrequencyCNN V2 architecture.
- Generates a 256-dimensional feature representation.

### 3.3 Feature Fusion and Classification

The extracted features from both branches are concatenated and passed to a fusion network.

| Component | Feature Dimension |
|---|---:|
| Xception RGB Features | 2048 |
| FrequencyCNN V2 Features | 256 |
| Combined Feature Vector | 2304 |

The fusion head processes the combined representation and predicts whether the input image is real or fake.

## 4. Model Architecture

```text
                 Input Facial Image
                         |
              +----------+----------+
              |                     |
              v                     v
         RGB Image             FFT Preprocessing
              |                     |
              v                     v
         Xception              Frequency CNN V2
              |                     |
              v                     v
       2048-D Features          256-D Features
              |                     |
              +----------+----------+
                         |
                         v
                 Feature Concatenation
                         |
                         v
                  2304-D Features
                         |
                         v
                    Fusion Head
                         |
                         v
                  Real / Fake
                 Classification
```

## 5. Datasets

The research uses the following datasets for training and evaluation.

### 5.1 FaceForensics++

FaceForensics++ is a widely used benchmark dataset for facial manipulation detection. It contains authentic videos and manipulated videos generated using different manipulation techniques.

**Dataset Source:**  
[FaceForensics++ C23 – Kaggle](https://www.kaggle.com/datasets/xdxd003/ff-c23)

### 5.2 WildDeepfake

WildDeepfake contains real-world deepfake videos collected from the internet. It provides a more challenging evaluation environment due to variations in video quality, compression, facial appearance, and manipulation methods.

**Dataset Source:**  
[WildDeepfake – Kaggle](https://www.kaggle.com/datasets/maysuni/wild-deepfake)

Both datasets are used to investigate the performance of the proposed architecture under different data distributions.

## 6. Repository Structure

```text
Research_Paper/
│
├── FaceForensics/
│   ├── xception/
│   ├── frequency/
│   ├── fusion/
│   ├── evaluation/
│   └── README.md
│
├── WildDeepfake/
│   ├── XCeption/
│   ├── Frequency/
│   ├── Fusion/
│   ├── Result/
│   └── README.md
│
└── README.md
```

Each dataset directory contains its respective training and evaluation implementation. Refer to the individual README files for dataset-specific instructions and execution details.

## 7. Experimental Evaluation

The proposed framework is evaluated using multiple metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC
- Confusion Matrix

### 7.1 WildDeepfake – In-Dataset Fusion V2 Evaluation

The Fusion V2 model was evaluated on a balanced test set of 6,768 images.

| Metric | Result |
|---|---:|
| Accuracy | 79.73% |
| Balanced Accuracy | 79.73% |
| Precision | 79.85% |
| Recall | 79.75% |
| F1-Score | 79.80% |
| Specificity | 79.70% |
| ROC-AUC | 88.51% |

The model achieved an ROC-AUC of 88.51% on the WildDeepfake test set, demonstrating its ability to distinguish real and manipulated images within the same dataset distribution.

### 7.2 Cross-Dataset Evaluation

Cross-dataset evaluation was conducted to investigate how well models trained on one dataset perform when tested on another dataset without target-dataset training or threshold tuning.

| Training Dataset / Model | Evaluation Dataset | Accuracy | ROC-AUC |
|---|---|---:|---:|
| FF++ Xception | WildDeepfake | 49.98% | 54.24% |
| WildDeepfake Xception | FF++ | 55.56% | 64.90% |
| WildDeepfake Fusion V2 | FF++ | 44.25% | 56.39% |

The cross-dataset experiments indicate a significant performance drop compared with in-dataset evaluation. This highlights the challenges associated with domain shifts, variations in compression, image quality, and manipulation characteristics across datasets.

These findings emphasize that strong in-dataset performance does not necessarily guarantee reliable generalization to unseen datasets.

*Note: Cross-dataset results are reported using their respective evaluation protocols. In particular, some evaluations use video-level aggregation, while others use image-level predictions.*

## 8. Technologies and Frameworks

The project uses the following technologies:

- **Python** – Primary programming language
- **PyTorch** – Deep learning framework
- **Torchvision** – Model and image processing utilities
- **timm** – Pretrained model support
- **OpenCV** – Image and video processing
- **NumPy** – Numerical computation
- **Pandas** – Dataset and experiment management
- **Scikit-learn** – Evaluation metrics and analysis
- **Matplotlib** – Result visualization

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yash-ch17/Research_Paper.git
cd Research_Paper
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the dependencies required by the respective dataset implementation.

```bash
pip install torch torchvision
pip install timm opencv-python numpy pandas scikit-learn matplotlib
```

### Step 4: Download the Datasets

Download the datasets from their respective sources:

- [FaceForensics++ C23](https://www.kaggle.com/datasets/xdxd003/ff-c23)
- [WildDeepfake](https://www.kaggle.com/datasets/maysuni/wild-deepfake)

Extract the datasets and organize them according to the instructions provided in the respective dataset README files.

### Step 5: Run Training and Evaluation

Navigate to the relevant dataset directory and follow its README for the exact training and evaluation commands.

- [FaceForensics README](FaceForensics/README.md)
- [WildDeepfake README](WildDeepfake/README.md)

## 10. Research Limitations

The experiments reveal several challenges:

- Cross-dataset performance remains considerably lower than in-dataset performance.
- Differences in compression, image quality, and manipulation techniques affect model predictions.
- Dataset-specific visual characteristics can influence learned representations.
- A model performing well on one benchmark may not generalize effectively to unseen data distributions.

These limitations motivate further investigation into domain generalization and robust feature learning.

## 11. Future Scope

The project provides several opportunities for further research and improvement.

### 11.1 Improving Cross-Dataset Generalization

A major future objective is to improve model performance across different datasets and unseen domains.

Potential directions include:

- Training with multiple datasets to increase data diversity.
- Exploring domain adaptation and domain generalization techniques.
- Improving frequency-domain feature representations.
- Investigating advanced feature fusion strategies.
- Developing more robust preprocessing and augmentation techniques.

### 11.2 Explainable Artificial Intelligence (XAI)

Explainability can be incorporated to make deepfake detection decisions more interpretable.

Potential future work includes:

- Integrating Grad-CAM for visualizing important image regions.
- Understanding which facial regions influence model predictions.
- Comparing spatial-domain and frequency-domain contributions.
- Improving transparency and interpretability of model decisions.

*Explainable AI is a proposed future enhancement and is not part of the current implementation.*

### 11.3 Model and Architecture Improvements

Further improvements may include:

- Exploring transformer-based architectures.
- Investigating attention-based feature fusion.
- Improving temporal information modeling for video-level detection.
- Evaluating robustness against unseen manipulation techniques.

## 12. Conclusion

This research investigates a hybrid deepfake detection framework that combines spatial-domain and frequency-domain feature learning through Xception and FrequencyCNN V2.

The proposed architecture integrates both feature representations using a fusion network and is evaluated on FaceForensics++ and WildDeepfake.

While the model demonstrates promising in-dataset performance, cross-dataset experiments highlight the continuing challenge of generalization across different data distributions.

The findings provide a foundation for further research into robust deepfake detection, improved cross-dataset generalization, and explainable AI techniques.

---

## Repository

**GitHub:** [Research_Paper](https://github.com/yash-ch17/Research_Paper)

**Research Focus:** Deepfake Detection | Frequency-Domain Learning | Feature Fusion | Cross-Dataset Generalization
