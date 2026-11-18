# 🌿 Plant Disease Classification with CNN & Open-Set Rejection

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a Convolutional Neural Network (CNN) built from scratch in Python, equipped with an **Out-of-Distribution (OOD) rejection system** to safely handle images from outside the dataset.

> **Project status:** Training, validation, evaluation, and OOD rejection pipeline are fully implemented and tested. The model achieves **90.68% validation accuracy** (at best epoch 18) and effectively rejects unknown/external images using embedding-based k-NN thresholding.

---

## 1. Problem

Plant diseases cause major crop losses, and diagnosing them manually requires expertise and time that farmers do not always have. Furthermore, standard classifiers suffer from the "forced-prediction" flaw: they classify any arbitrary image (e.g., objects, random photos, non-leaf items) into one of the trained classes. 

The goals are:
- Accurately determine the plant species and specific disease from a leaf photo.
- **Reject external/unknown images** that do not belong to any of the 38 trained classes.

## 2. Solution

- Build a **CNN** from scratch to automatically learn disease features (spots, discoloration, deformities).
- Implement **Class Weights** to handle severe class imbalance.
- Implement an **Open-Set Rejection Pipeline** using feature embeddings (Dense 256 layer) and cosine similarity k-NN scoring against a training reference set.

## 3. Dataset

[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

| Item | Value |
|---|---|
| Images (color version) | 54,305 |
| Classes | 38 |
| Image size | 256×256 |
| Smallest class | `Potato___healthy` (152 images) |
| Largest class | `Orange___Haunglongbing_(Citrus_greening)` (5,507 images) |

---

## 4. Project Workflow & Training Highlights

### Data Cleaning & Splitting
- Inspected 54,305 images in parallel, checked for corruption and dimensions.
- Removed **21 duplicate/conflicting images**, leaving **54,284 clean images** (`clean_df.csv`).
- Stratified split: **70% train / 15% validation / 15% test**.

### Handling Class Imbalance (Class Weights)
- Computed balanced class weights (`compute_class_weight`) to penalize mistakes on rare classes more heavily.
- **Note on trade-off:** Documented in evaluations that weighting successfully raised the recall of small classes, though it slightly impacted their precision as the model became more eager to predict them.

### Training & Early Stopping Results
- Trained for up to 30 epochs with `Adam optimizer`, `ReduceLROnPlateau`, and `EarlyStopping`.
- **Training stopped at Epoch 23** due to EarlyStopping (`patience=5`).
- **Best Epoch:** **Epoch 18**, achieving:
  - **Best `val_accuracy`:** `90.68%`
  - **Best `val_loss`:** `0.2896`
  - **Train accuracy at best epoch:** `94.3%` (under augmentation)

### Out-of-Distribution (OOD) / Open-Set Rejection Mechanism
To prevent the model from blindly classifying non-plant or external images:
1. Extracted 256-dimensional feature embeddings from the trained model's dense layer.
2. Built a reference pool (`ref`) of 10,000 L2-normalized training embeddings.
3. Computed cosine similarity and top-5 k-NN average score for validation and external images.
4. Established a threshold (`thr_sim`) targeting a 95% acceptance rate on validation data. Images falling below this threshold are flagged and rejected as **"غير معروف" (Unknown)**.

---

## 5. Tech Stack

- **Python**, **TensorFlow / Keras** (`tf.data`, custom CNN)
- **Pandas / NumPy**
- **scikit-learn** (metrics, class weights, train-test split)
- **Pillow (PIL)** (image inspection & processing)
- **Matplotlib**

## 6. Getting Started (Google Colab)

1. Open the notebook in Colab and enable GPU acceleration.
2. Store your Kaggle API token in Colab Secrets as `KAGGLE_API_TOKEN`.
3. Run pipeline cells sequentially to download, clean, split, train, and deploy the OOD rejection system.

```bash
pip install kaggle pillow pandas scikit-learn matplotlib tqdm tensorflow bing-image-downloader
