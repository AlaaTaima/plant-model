# 🌿 Plant Disease Classification with CNN

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a Convolutional Neural Network (CNN) built in Python.

> **Project status:** Work in progress. Data preparation, cleaning, and splitting are complete, and the image loading pipeline and model definition are built. Training and evaluation are underway.

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
- **TensorFlow / Keras**: model building and training, and the image loading pipeline (`tf.data`)
- **Pandas / NumPy**: dataset table and data processing
- **Pillow (PIL)**: reading and validating images
- **scikit-learn**: dataset splitting, class weights, and evaluation metrics
- **Matplotlib**: plots
- **hashlib (MD5)**: exact duplicate detection
- **concurrent.futures (ThreadPoolExecutor)**: parallel file inspection
- **Kaggle API**: downloading the dataset from Kaggle
- **Google Colab**: runtime environment (with GPU), and **Google Drive** for storing tables and the model

## 5. Project Workflow

### Stage 1: Data preparation and inspection

- Download the dataset from Kaggle through the API directly onto Colab's local disk, extracting only the `color` folder (reading from local disk is much faster than from Drive).
- Build a **DataFrame** with one row per image, inspected in parallel:

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
| Color mode | 54,304 RGB images and one RGBA image (its alpha channel is fully opaque, so it is converted to RGB at load time) |
| Duplicate images (identical bytes) | **21** images, **removed** |
| Images appearing under more than one class | **0** |

**Outcome:** from 54,305 images to **54,284 clean images**. No files were deleted from the source. The table was filtered instead, and the clean table (`clean_df.csv`) was saved to Drive. The table is sorted by path so the result is identical across sessions.

The class distribution is also **imbalanced** (about a 36× gap between the smallest and largest class), which is handled during training with class weights.

### Stage 3: Dataset splitting

- **Stratified** split into **70% train / 15% validation / 15% test** using `train_test_split` with `stratify` and `random_state=42`, so class proportions stay similar in every subset.
- The splits are saved as fixed files (`train.csv`, `val.csv`, `test.csv`), so the split does not change between sessions and the test set stays untouched until the final evaluation.
- **Note:** the images were taken under lab conditions with a uniform background and may include close shots of the same leaf, so test accuracy may be slightly optimistic. Testing the model on images from outside the dataset is recommended.

### Stage 4: Image pipeline and model

**Image pipeline (`tf.data`):**
- Read each file, decode it to 3 RGB channels, and resize to **128×128**.
- Shuffle training data only, then batch (64) and `prefetch` for faster training.
- Normalization (÷255) happens inside the model itself.

**CNN architecture (built from scratch):**

```
Input 128×128×3
  → Data Augmentation (flip, rotation, zoom, contrast)
  → Rescaling (1/255)
  → 4 × [ Conv → BatchNorm → Conv → BatchNorm → MaxPool → Dropout(0.25) ]
        (filters: 32 → 64 → 128 → 256)
  → GlobalAveragePooling
  → Dense(256) + Dropout(0.5)
  → Dense(38, Softmax)
```

**Training setup:**
- Adam optimizer with a learning rate of 0.001, and `sparse_categorical_crossentropy` loss.
- **Class imbalance:** `class_weight` computed from the training set only.
- Callbacks: `ModelCheckpoint` (save the best model), `EarlyStopping`, and `ReduceLROnPlateau`.
- Up to 30 epochs, with GPU enabled.

#### Handling class imbalance (Class Weights)

**Problem:** the number of images varies a lot between classes (from 152 to 5,507, about 36×), so the model sees large classes thousands of times and small classes only a few times, and learns the large ones better. Overall accuracy alone is misleading here: a class such as `Potato___healthy` is under 0.3% of the data, so even if the model got every one of its images wrong, overall accuracy would drop by only that much.

**Solution:** multiply each image's loss by a weight based on its class, so a mistake on a rare-class image costs more than a mistake on a common-class image:

```
weight = number of training images / (number of classes × images in the class)
```

Rough example: the smallest class (~107 training images) gets a weight near 9, and the largest (~3,850 images) a weight near 0.26, so each class contributes roughly equally to training.

```python
w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_df["y"].values)
class_weight = dict(enumerate(w))
cnn.fit(train_ds, validation_data=val_ds, class_weight=class_weight, ...)
```

- Weights are computed from the **training set only**.
- They work together with the data augmentation layers inside the model, which add variety to small-class images.
- **Measuring success:** not overall accuracy alone, but **macro F1** and **per-class recall** on the validation set.
- **Fallback:** if some small classes stay weak, try oversampling (repeating small-class images with augmentation) **instead of** class_weight, not together with it, and compare by macro F1 on validation.
- SMOTE was not used because it is designed for tabular data, and interpolating pixels between two images produces an unrealistic image.

### Stage 5: Evaluation

- Decisions and comparisons are made on the **validation** set (accuracy, macro F1, and a per-class report).
- The **test** set is used **once**, at the end: accuracy, plus precision, recall, and F1 per class.
- Try the model on images from outside the dataset to measure real-world performance.

## 6. Roadmap

- [x] Download the dataset and inspect its structure and class distribution
- [x] Build the image table and check for corrupted files
- [x] Detect and remove duplicate images (21 images)
- [x] Split the dataset (train / val / test)
- [x] Build the image loading pipeline
- [x] Define the CNN architecture
- [ ] Train the model with `class_weight`
- [ ] Evaluate on validation, then on test
- [ ] Error analysis and testing on images from outside the dataset

## 7. Getting Started

The project is written for **Google Colab**:

1. Open the notebook in Colab and enable a GPU (Runtime → Change runtime type).
2. Create a Kaggle API token and store it in **Colab Secrets** as `KAGGLE_API_TOKEN` (never hardcode it in the notebook or push it to GitHub).
3. Run the cells in order (do not use Run all): download and load, then build the table, clean, and split (once only), then model and training.
4. In any later session, read `train.csv`, `val.csv`, and `test.csv` from Drive instead of splitting again.

```bash
pip install kaggle pillow pandas scikit-learn matplotlib tqdm tensorflow
```
