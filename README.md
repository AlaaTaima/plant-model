# 🌿 Plant Disease Classification with CNN

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a **Convolutional Neural Network (CNN) built from scratch** with TensorFlow / Keras.

> **Project status:** Work in progress. Data preparation, cleaning, and leakage-safe splitting are complete, and the input pipeline and CNN architecture are built. Training and evaluation are next.

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
| Train | 38,777 | 71.4% |
| Validation | 7,760 | 14.3% |
| Test | 7,747 | 14.3% |

- Group overlap between train/val, train/test, and val/test: **0 / 0 / 0**.
- Smallest class in the test set: 22 images.
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

**Training results (validation set):** _to be added after training._

### Stage 5: Evaluation (upcoming)

- Accuracy, per-class precision / recall / F1, and a confusion matrix on the **test set** (evaluated once, after all tuning is finished).
- Error analysis of the weakest classes.
- Test the model on images from outside the dataset to measure real-world performance.

## 6. Roadmap

- [x] Download the dataset and inspect its structure and class distribution
- [x] Build the image table and check for corrupted files
- [x] Detect and remove duplicate images (21 images)
- [x] Handle data leakage with perceptual hashing (pHash, 8 orientations)
- [x] Split the dataset (train / val / test) with `StratifiedGroupKFold`
- [x] Build the image loading pipeline (`tf.data`)
- [x] Train the CNN
- [x] Final evaluation on the test set and error analysis
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
