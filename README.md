# cnn-pneumoniaMNIST

# PneumoniaMNIST Binary Classification Pipeline

A modular, lightweight Deep Learning pipeline using a Convolutional Neural Network (CNN) to detect pneumonia from pediatric chest X-ray images provided by the **PneumoniaMNIST** dataset (part of the MedMNIST v2 collection).

---

## Project Overview

Pneumonia is a life-threatening inflammatory condition of the lungs. Automated diagnosis via chest X-rays can assist healthcare professionals in quick triage.

This repository implements an end-to-end classification pipeline built with **TensorFlow / Keras**. The codebase is explicitly decoupled into independent script stages:

1. **Data Preparation**: Downloads, normalizes, and splits image arrays into disk storage (`.npy`).
2. **Model Definition**: Modular CNN architecture setup.
3. **Training Routine**: Configured with early stopping, dynamic learning rate adjustment, and check-pointing.
4. **Evaluation**: Evaluates performance on unseen test data and exports metrics and a confusion matrix.

---

## Dataset Details

- **Dataset**: PneumoniaMNIST (MedMNIST v2)
- **Image Dimensions**: $28 \times 28$ pixels (Grayscale, 1 Channel)
- **Classes**: 2 (`0`: Normal, `1`: Pneumonia)
- **Data Splits**:
  - **Train**: 4,708 samples
  - **Validation**: 524 samples
  - **Test**: 624 samples

---

## Repository Structure

```plaintext
.
├── .gitignore          # Prevents committing datasets, models, and virtualenvs
├── requirements.txt    # Python dependencies list
├── data_prep.py        # Stage 1: Downloads and converts data to data/*.npy
├── model.py            # Stage 2: Defines the CNN architecture
├── train.py            # Stage 3: Trains model and saves best_model.h5
├── evaluate.py         # Stage 4: Generates evaluation metrics & confusion_matrix.png
└── data/               # Local directory containing generated .npy files (Git-ignored)
```

---

## Installation & Setup

1. **Clone the repository** (or navigate to your project directory):

   ```bash
   cd cnn-pneumonia-medmnist
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv

   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Pipeline

Execute the python scripts sequentially:

### Step 1: Prepare & Process Data

Downloads PneumoniaMNIST, scales pixel intensities to $[0, 1]$, and saves NumPy binaries under `data/`:

```bash
python data_prep.py
```

### Step 2: Inspect Model Architecture

Verifies layer shapes and parameter counts:

```bash
python model.py
```

### Step 3: Train the CNN

Trains the neural network using `Adam` optimizer and `binary_crossentropy` loss. Saves optimal weights to `best_model.h5`:

```bash
python train.py
```

### Step 4: Evaluate Test Performance

Evaluates `best_model.h5` against the test set, printing Precision, Recall (Sensitivity), F1-Score, and saving `confusion_matrix.png`:

```bash
python evaluate.py
```

---

## Key Clinical Considerations

- **Primary Evaluation Metric (Recall):** In medical diagnostic tasks, **False Negatives** (misclassifying a sick patient as Normal) carry severe risks. Thus, model tuning prioritizes high **Recall (Sensitivity)** over raw accuracy.
- **Data Augmentation Constraints:** Standard horizontal flipping is **not** applied to chest X-rays because anatomical orientation (e.g., heart location on the left) carries diagnostic relevance.

---
