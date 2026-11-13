# 🌿 Plant Disease Classification with CNN

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a **Convolutional Neural Network (CNN) built from scratch** with TensorFlow / Keras.

> **Project status:** The CNN is trained and evaluated (**94.6% test accuracy**). A FastAPI web service for the model is built; deployment and error analysis are next.

---

## 1. Problem

Plant diseases cause major crop losses, and diagnosing them manually requires expertise and time that farmers do not always have.

The goal is a model that takes a **photo of a plant leaf** and determines:
- The plant species.
- Whether the leaf is healthy or diseased, and which disease it has.

## 2. Solution

Build a **CNN** for image classification that automatically learns disease features from images (spots, discoloration, deformities) instead of relying on hand-crafted rules. The network is designed and trained from scratch (no pretrained weights).

Output: a probability for each of the 38 classes. The class with the highest probability is the diagnosis.

## 3. Dataset

[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

| Item | Value |
|---|---|
| Images (color version) | 54,305 |
| Classes | 38 |
| Image size | 256×256 |
| Smallest class | `Potato___healthy` (152 images) |
| Largest class | `Orange___Haunglongbing_(Citrus_greening)` (5,507 images) |

The dataset ships in three versions of the same images (`color`, `grayscale`, `segmented`). Only the **color** version is used, so the same images are not repeated in training.

## 4. Tech Stack

- **Python**
- **TensorFlow / Keras**: model building and training (`tf.data` input pipeline)
- **Pandas / NumPy**: dataset table and data processing
- **Pillow (PIL)**: reading and validating images
- **imagehash + SciPy**: perceptual hashing and grouping of near-duplicate images
- **scikit-learn**: grouped, stratified splitting and evaluation metrics
- **Matplotlib**: plots
- **hashlib (MD5)**: exact duplicate detection
- **Google Colab (T4 GPU)**: runtime environment, with **Google Drive** for storing tables and checkpoints
- **Kaggle API**: downloading the dataset directly to Colab's local disk

## 5. Project Workflow

### Stage 1: Data preparation and inspection

- Download the dataset from Kaggle straight to Colab's local disk and work with the `color` version.
  Copying ~54k small files from Drive was far too slow, so the dataset is downloaded from Kaggle in each new session instead.
- Build a **DataFrame** with one row per image:

| Column | Description |
|---|---|
| `path` | Image path |
| `label` | Class (folder name) |
| `fname` | File name |
| `bytes` | File size |
| `width` / `height` / `mode` | Image dimensions and color mode |
| `corrupted` | Whether the file is corrupted |
| `hash` | MD5 hash of the file |

### Stage 2: Cleaning

Since the data is images rather than tables, "null values" correspond to corrupted or invalid files. Inspection results:

| Check | Result |
|---|---|
| Corrupted or unreadable files | **0** |
| Zero-byte files | **0** |
| Image dimensions | **256×256** for all images |
| Duplicate images (identical bytes) | **21** images, **removed** |
| Images appearing under more than one class | **0** |

**Outcome:** from 54,305 images to **54,284 clean images**. No files were deleted from the source. The table was filtered instead, and the clean table was saved to Drive.

The class distribution is **imbalanced** (about a 36× gap between the smallest and largest class). It is handled with a **stratified** split and data augmentation. Class weights were not used.

### Stage 3: Dataset splitting (leakage-safe)

The dataset contains modified copies (flipped, rotated, slightly different shots) of the same leaf. If one copy lands in train and another in test, accuracy looks higher than it really is.

**Method:**
1. Compute a **perceptual hash (pHash)** for each image in **8 orientations** (4 rotations × horizontal flip), so rotated or flipped copies are still matched.
2. Compare hashes **within each class** and link images whose minimum Hamming distance is **≤ 2** (out of 64 bits).
3. Merge linked images into **groups** using connected components.
4. Split with `StratifiedGroupKFold` so that no group is spread across more than one subset.

**Threshold tuning:** a threshold of 4 caused "chaining" (the largest group reached 1,133 unrelated images), so it was lowered to 2.

| Grouping result (threshold = 2) | Value |
|---|---|
| Total groups | 54,019 |
| Groups with more than one image | 159 |
| Largest group | 27 images |

**Final split:**

| Subset | Images | Share |
|---|---|---|
| Train | 38,792 | 71.5% |
| Validation | 7,754 | 14.3% |
| Test | 7,738 | 14.3% |

- Group overlap between train/val, train/test, and val/test: **0 / 0 / 0**.
- Smallest class in the test set: 22 images (`Potato___healthy`).
- The split tables are saved as `train.csv`, `val.csv`, and `test.csv`.

### Stage 4: Model building and training

**Input pipeline (`tf.data`):** read JPEG → decode → resize to **128×128** → batch of 64 → prefetch. Pixel values are scaled to [0, 1] inside the model with a `Rescaling(1/255)` layer.

**Architecture (CNN from scratch):**

```
Image → Augmentation → Rescaling (÷255)
      → [Conv3×3 → BN → Conv3×3 → BN → MaxPool → Dropout(0.25)] × 4   (filters: 32, 64, 128, 256)
      → GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(38, softmax)
```

- **Augmentation (training only):** random flip (horizontal + vertical), rotation (±15%), zoom (10%), contrast (10%).
- **BatchNormalization** after each convolution for stable and faster training.
- **Dropout** and **GlobalAveragePooling** to reduce overfitting.

**Training setup:**

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 |
| Loss | Sparse categorical cross-entropy |
| Max epochs | 30 |
| Batch size | 64 |
| `ModelCheckpoint` | Saves the best model by `val_accuracy` to Drive (`cnn_best.keras`) |
| `EarlyStopping` | Patience 5, restores the best weights |
| `ReduceLROnPlateau` | Halves the learning rate after 2 epochs without improvement |

**Training results (validation set):**

Training stopped automatically at epoch 16 (early stopping). The best epoch was **epoch 11**, and its weights were restored.

| | Epoch 11 (best) | Epoch 16 (last) |
|---|---|---|
| Train accuracy | 97.6% | 98.7% |
| Validation accuracy | **94.3%** | 93.7% |
| Validation loss | 0.183 | 0.213 |

Validation accuracy fluctuated noticeably between epochs (from about 56% up to 94%) while training accuracy rose steadily. This is common for a from-scratch CNN with BatchNormalization at a relatively high learning rate. The gap between training and validation accuracy at the end indicates mild overfitting.

### Stage 5: Evaluation on the test set

The test set was evaluated once, with the best checkpoint (`cnn_best.keras`).

| Metric | Value |
|---|---|
| **Test accuracy** | **94.6%** |
| Macro F1 | 0.935 |
| Weighted F1 | 0.946 |

Test accuracy (94.6%) is consistent with validation accuracy (94.3%), so the validation set was not overfitted.

**Strongest classes (F1 ≥ 0.99):** `Corn_(maize)___healthy` (1.000), `Corn_(maize)___Common_rust_` (0.997), `Strawberry___Leaf_scorch` (0.991), `Tomato___Tomato_mosaic_virus` (0.991), `Orange___Haunglongbing_(Citrus_greening)` (0.990).

**Weakest classes (lowest F1):**

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| `Potato___healthy` | 0.895 | 0.773 | 0.829 | 22 |
| `Raspberry___healthy` | 0.875 | 0.808 | 0.840 | 52 |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 0.957 | 0.748 | 0.840 | 238 |
| `Pepper,_bell___healthy` | 1.000 | 0.776 | 0.874 | 210 |
| `Strawberry___healthy` | 0.947 | 0.831 | 0.885 | 65 |
| `Apple___healthy` | 0.801 | 0.987 | 0.885 | 233 |

**Observations:**
- Several "healthy" classes and a few small classes are the weakest. Some classes (`Apple___healthy`, `Tomato___healthy`) have high recall but lower precision, meaning other images are being assigned to them. Others (`Pepper,_bell___healthy`, `Tomato___Spider_mites`) have high precision but lower recall, meaning their images are being assigned elsewhere.
- Very small classes such as `Potato___healthy` (22 test images) are sensitive to a single error.

**Planned:** a confusion-matrix analysis of the most frequent mistakes, and testing on images from outside the dataset to measure real-world performance (PlantVillage images are lab-style photos with plain backgrounds, so accuracy on field photos is expected to be lower).

## 6. Roadmap

- [x] Download the dataset and inspect its structure and class distribution
- [x] Build the image table and check for corrupted files
- [x] Detect and remove duplicate images (21 images)
- [x] Handle data leakage with perceptual hashing (pHash, 8 orientations)
- [x] Split the dataset (train / val / test) with `StratifiedGroupKFold`
- [x] Build the image loading pipeline (`tf.data`)
- [x] Train the CNN (best validation accuracy 94.3%)
- [x] Final evaluation on the test set (accuracy 94.6%)
- [x] Build a FastAPI service and web interface for the model
- [ ] Deploy the service (Render)
- [ ] Confusion-matrix error analysis
- [ ] Test on real-world photos from outside the dataset

## 7. Getting Started

The project is written for **Google Colab** (use a **T4 GPU** runtime):

1. Open the notebook in Colab and set Runtime → Change runtime type → T4 GPU.
2. Mount Google Drive.
3. Download the dataset from Kaggle (API token required) and unzip only the `color` folder to `/content`.
4. Load `train.csv`, `val.csv`, and `test.csv` from Drive.
5. Run the cells in order (do not use Run all).

```bash
pip install "kaggle>=1.8.0" imagehash pillow pandas scikit-learn scipy matplotlib tensorflow
```

> The `/content` folder is deleted when the Colab session ends, so the dataset must be downloaded again in each new session. The tables and the best model checkpoint (`cnn_best.keras`) are kept in Google Drive.
