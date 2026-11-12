# 🌿 Plant Disease Classification with CNN

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a Convolutional Neural Network (CNN) built in Python.

> **Project status:** Work in progress. Data preparation and cleaning are complete. Dataset splitting and data-leakage handling are underway, followed by model building and training.

---

## 1. Problem

Plant diseases cause major crop losses, and diagnosing them manually requires expertise and time that farmers do not always have.

The goal is a model that takes a **photo of a plant leaf** and determines:
- The plant species.
- Whether the leaf is healthy or diseased, and which disease it has.

## 2. Solution

Build a **CNN** for image classification that automatically learns disease features from images (spots, discoloration, deformities) instead of relying on hand-crafted rules.

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
- **TensorFlow / Keras**: model building and training
- **Pandas / NumPy**: dataset table and data processing
- **Pillow (PIL)**: reading and validating images
- **scikit-learn**: dataset splitting and class weights
- **Matplotlib**: plots
- **hashlib (MD5)**: exact duplicate detection
- **Google Colab**: runtime environment, with **Google Drive** for storing data and results
- **kagglehub**: downloading the dataset from Kaggle

## 5. Project Workflow

### Stage 1: Data preparation and inspection

- Download the dataset from Kaggle and work with the `color` version.
- Copy the images to Colab's local disk to speed up reading (reading directly from Drive is slow).
- Build a **DataFrame** with one row per image:

| Column | Description |
|---|---|
| `path` | Image path |
| `label` | Class (folder name) |
| `ext` | File extension |
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
| Color mode | 54,304 RGB images and one RGBA image (its alpha channel is fully opaque, so it is converted to RGB at load time) |
| Duplicate images (identical bytes) | **21** images, **removed** |
| Images appearing under more than one class | **0** |

**Outcome:** from 54,305 images to **54,284 clean images**. No files were deleted from the source. The table was filtered instead, and the clean table was saved to Drive.

The class distribution is also **imbalanced** (about a 36× gap between the smallest and largest class). This will be addressed with class weights and data augmentation.

### Stage 3: Dataset splitting (in progress)

- **Stratified** split into roughly **70% train / 15% validation / 15% test**, preserving class proportions in each subset.
- **Data leakage:** the dataset may contain modified copies (flipped, rotated, or brightness-adjusted) of the same image. If one copy lands in train and another in test, accuracy looks higher than it really is.
- **Planned solution:** compute a **perceptual hash (dHash)** for each image across eight orientations (rotations and flips), group visually similar images together, then split with `StratifiedGroupKFold` so that no group is spread across more than one subset.

### Stage 4: Model building and training (upcoming)

```
Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Softmax (38 classes)
```

- Load images, convert to RGB, resize (e.g. 128×128), and normalize (÷255).
- Data augmentation for training only.
- `class_weight` to handle class imbalance.

### Stage 5: Evaluation (upcoming)

- Accuracy, per-class F1, and a confusion matrix on the test set.
- Test the model on images from outside the dataset to measure real-world performance.

## 6. Roadmap

- [x] Download the dataset and inspect its structure and class distribution
- [x] Build the image table and check for corrupted files
- [x] Detect and remove duplicate images (21 images)
- [ ] Handle data leakage with perceptual hashing
- [ ] Split the dataset (train / val / test)
- [ ] Build the image loading pipeline
- [ ] Build and train the CNN model
- [ ] Evaluation and error analysis

## 7. Getting Started

The project is written for **Google Colab**:

1. Open the notebook in Colab.
2. Mount Google Drive and download the dataset with `kagglehub`.
3. Run the cells in order (do not use Run all).

```bash
pip install kagglehub pillow pandas scikit-learn matplotlib tensorflow
```
