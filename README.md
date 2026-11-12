# 🌿 Plant Disease Classification with EfficientNetB0

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using **transfer learning** with EfficientNetB0 (TensorFlow / Keras).

> **Project status:** Work in progress. Data preparation, cleaning, leakage-safe splitting, and two-stage training are complete (validation accuracy ≈ **97.7%**). Final evaluation on the test set is next.

---

## 1. Problem

Plant diseases cause major crop losses, and diagnosing them manually requires expertise and time that farmers do not always have.

The goal is a model that takes a **photo of a plant leaf** and determines:
- The plant species.
- Whether the leaf is healthy or diseased, and which disease it has.

## 2. Solution

Fine-tune a pretrained **EfficientNetB0** (ImageNet weights) for image classification. Instead of training a CNN from scratch, the model reuses features learned from millions of images and adapts them to leaf diseases.

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

**Input pipeline (`tf.data`):** read JPEG → decode → resize to **224×224** → batch of 64 → prefetch. EfficientNet normalizes inputs internally, so there is no manual ÷255.

**Architecture:**

```
Image → Augmentation → EfficientNetB0 (ImageNet) → GlobalAveragePooling → Dropout(0.3) → Dense(38, softmax)
```

- **Augmentation (training only):** random flip (horizontal + vertical), rotation (±15%), zoom (10%), contrast (10%).
- Total parameters: 4,098,249.

**Two-stage training:**

| | Stage 1: Train the head | Stage 2: Fine-tuning |
|---|---|---|
| Trainable layers | Final `Dense` layer only (48,678 params) | Head + last 30 layers of EfficientNetB0 |
| BatchNorm layers | Frozen | Frozen |
| Optimizer | Adam, lr = 1e-3 | Adam, lr = 1e-5 |
| Epochs | 8 | 10 |
| Loss | Sparse categorical cross-entropy | Sparse categorical cross-entropy |

Callbacks: `ModelCheckpoint` (saves the best model by `val_accuracy` to Drive) and `EarlyStopping` (patience 3, restores best weights).

**Training results (validation set):**

| Stage | Train accuracy | Val accuracy | Val loss |
|---|---|---|---|
| Stage 1 (epoch 8) | 94.4% | **95.6%** | 0.143 |
| Stage 2 (epoch 10) | 97.4% | **97.7%** | 0.069 |

Validation accuracy is slightly above training accuracy because augmentation and dropout are active only during training. There is no sign of overfitting, and the validation loss was still decreasing at the end of Stage 2.

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
- [x] Build and train the model (EfficientNetB0, two-stage transfer learning)
- [ ] Optional: continue fine-tuning with more unfrozen layers
- [ ] Final evaluation on the test set and error analysis
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

> The `/content` folder is deleted when the Colab session ends, so the dataset must be downloaded again in each new session. The tables and the best model checkpoint (`best.keras`) are kept in Google Drive.
