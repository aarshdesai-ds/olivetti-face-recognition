# 🧠 Augmented Olivetti Face Recognition

This project performs face recognition using the **Augmented Olivetti Faces Dataset**, applying both supervised and unsupervised machine learning techniques to classify facial identities and explore visual structure.

We use:
- ✅ **Support Vector Machine (SVM)** for high-performance face classification
- 👥 **K-Nearest Neighbors (KNN)** as a baseline model
- 🔍 **K-Means Clustering** to group faces without labels
- 📉 **Principal Component Analysis (PCA)** for dimensionality reduction and 2D visualization

---

## 📦 Dataset Overview

**Source**: [Kaggle – Augmented Olivetti Faces Dataset](https://www.kaggle.com/datasets/martininf1n1ty/olivetti-faces-augmented-dataset)

The dataset includes:
- 2,000 grayscale face images of 40 individuals (64x64 pixels each)
- Augmentations: flipped, rotated, noisy, and contrast-adjusted variants
- Provided in NumPy `.npy` format:
  - `augmented_faces.npy` – shape `(2000, 64, 64)`
  - `augmented_labels.npy` – shape `(2000,)`

---

## 🧠 Modeling Pipeline

### 📌 1. Preprocessing & Visualization
- Reshaped 64×64 images into 4096-dimensional vectors
- Merged features and labels into a structured DataFrame
- Displayed random and per-label sample faces

### 📉 2. PCA (Principal Component Analysis)
- Reduced feature space from 4096 → 2 dimensions for plotting
- Visualized class separation and cluster density

### 🤖 3. Supervised Learning

#### 🌟 Support Vector Machine (SVM)
- Linear kernel
- Training Accuracy: **100%**
- Test Accuracy: **98%**
- Precision/Recall: Near-perfect on all classes

#### 👥 K-Nearest Neighbors (KNN)
- K = 5
- Training Accuracy: **94.9%**
- Test Accuracy: **86.2%**
- Strong but slightly more sensitive to overlap between classes

### 🔍 4. Unsupervised Learning (K-Means)
- Performed on PCA-reduced data
- Clusters showed meaningful structure aligned with identities
- Some class overlap due to 2D projection

---

## 📊 Results Summary

| Model         | Train Accuracy | Test Accuracy | Notes                      |
|---------------|----------------|---------------|----------------------------|
| SVM (Linear)  | 100%           | **98%**       | Best performance overall   |
| KNN (k=5)     | 94.9%          | 86.2%         | Baseline, more variability |
| K-Means (PCA) | –              | –             | Shows structure, not labels|

---

## 📈 Visual Insights

- PCA plots revealed high separability between face classes even in 2D
- Confusion matrices showed class-level precision for both models
- K-Means clustered faces with reasonable grouping, supporting PCA’s effectiveness

---

## 🧰 Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (SVM, KNN, PCA, KMeans)

---

## 🚀 How to Run

1. Clone this repository  
   `git clone https://github.com/<your-username>/diamond-price-prediction.git`
2. Launch the notebook  
   `jupyter notebook face-recognition.ipynb`
3. Run all cells to reproduce the workflow and results

---



## ✅ Key Takeaways
SVM provided outstanding classification accuracy and generalization

KNN performed well but was slightly less robust to class similarity

PCA + KMeans revealed consistent visual identity clusters even without labels

This project shows traditional ML can achieve high performance in face recognition when paired with clean data and clear visualization
