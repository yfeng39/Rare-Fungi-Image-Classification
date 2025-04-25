# Rare Fungi Image Classification with Few-Shot Learning

## Project Overview

This project focuses on the **classification of rare fungi species** using **few-shot learning (FSL)** techniques. Traditional deep learning models struggle with rare species due to extremely limited available images. By applying state-of-the-art **metric-learning based few-shot methods** on the **Danish Fungi 2020 (DF20) dataset**, we aim to overcome the challenges posed by data scarcity and class imbalance in fungi image classification.

Our work experiments with multiple FSL methods, explores the impact of **data augmentation** and **self-supervised learning tasks**, and proposes effective strategies to boost performance in classifying rare fungi species.

---

## Key Features

- **Dataset:** Danish Fungi 2020 (DF20)  
  - 1,604 classes and ~296,000 images
  - Focused on rare species (classes with fewer than 40 images)
  
- **Few-Shot Learning Methods Tested:**  
  - ProtoNet
  - RelationNet
  - CovaMNet
  - DN4
  - CAN (Cross Attention Network)

- **Backbone Architectures:**  
  - ResNet18 (main backbone used)
  - Conv64F (tested for comparison)

- **Data Augmentation Techniques:**  
  - Brightness, Contrast, Saturation adjustment
  - Random Grayscale, Horizontal Flip

- **Self-Supervised Learning Tasks:**  
  - Jigsaw Puzzle Prediction
  - Rotation Angle Prediction

- **Experimental Setups:**  
  - 5-way 5-shot and 20-way 5-shot classification tasks
  - Testing the effect of individual and combined augmentations
  - Evaluating the integration of self-supervised tasks into few-shot pipelines

---

## Major Results

- **Best Performing Model:**  
  - **CAN (Cross Attention Network)** + Saturation Augmentation achieved **88.40% Top-1 Accuracy** on 5-way 5-shot tasks.

- **Findings on Augmentation:**  
  - Simple augmentations (e.g., Horizontal Flip) improved performance, but combining multiple augmentations often degraded results.

- **Findings on Self-Supervised Learning:**  
  - Self-supervised tasks improved some FSL methods (e.g., RelationNet, CovaMNet) but had mixed results overall.
  - Choice of the λ (loss weighting) hyperparameter critically influenced performance.

---

## How to Use

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/rare-fungi-fsl.git
cd rare-fungi-fsl
```

2. **Prepare Dataset:**
   - Obtain the DF20 dataset (publicly available) and organize it according to the required training/validation/testing splits.

3. **Install Requirements:**

```bash
pip install -r requirements.txt
```

4. **Train Models:**

```bash
python train.py --method CAN --backbone resnet18 --augment saturation --fsl_task 5way5shot
```

5. **Evaluate:**

```bash
python evaluate.py --checkpoint path/to/checkpoint --fsl_task 5way5shot
```

---

## Directory Structure

```
.
├── data/           # DF20 dataset organization
├── models/         # FSL models (ProtoNet, CAN, etc.)
├── augmentations/  # Data augmentation implementations
├── ssl/            # Self-supervised learning modules
├── train.py        # Training script
├── evaluate.py     # Evaluation script
└── README.md       # Project introduction
```

---

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{anonymous2025fungiFSL,
  title={Rare Fungi Image Classification Based on Few-Shot Learning},
  author={Anonymous},
  booktitle={IEEE ICME},
  year={2025}
}
```

---

## Acknowledgements

- Danish Fungi 2020 Dataset
- LibFewShot Library
- Previous works on few-shot learning and self-supervised techniques

