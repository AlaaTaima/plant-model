# 🌿 Plant Disease Classification with CNN

A deep learning project that classifies plant leaf images into **38 classes** (different diseases and healthy leaves) using a Convolutional Neural Network (CNN) built from scratch in Python.

> **Project status:** Training, validation, and test evaluation are complete. The model reaches **95.0% accuracy** and **0.945 macro F1** on the held-out test set. Remaining work: error analysis and handling images from outside the dataset.

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
- Resulting sizes: about 37,998 train, **8,143 validation**, and **8,143 test** images.
- **Note:** the images were taken under lab conditions with a uniform background and may include close shots of the same leaf, so test accuracy may be slightly optimistic. Testing the model on images from outside the dataset is recommended (see Stage 6).

### Stage 4: Image pipeline and model

**Image pipeline (`tf.data`):**
- Read each file, decode it to 3 RGB channels, and resize to **128×128**.
- Shuffle training data only, then batch (64) and `prefetch` for faster training.
- Normalization (÷255) happens inside the model itself, so the model expects raw pixel values in the range 0–255.

**CNN architecture (built from scratch, ~1.25M parameters):**

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

The augmentation layers are active only during training and are switched off automatically at prediction time.

### Stage 5: Training

**Setup:**
- Adam optimizer with an initial learning rate of 0.001, and `sparse_categorical_crossentropy` loss.
- Up to 30 epochs, batch size 64, GPU enabled (~128 seconds per epoch).
- **Class imbalance:** `class_weight` computed from the training set only (explained below).
- Three callbacks:

```python
cb = [
    keras.callbacks.ModelCheckpoint(".../cnn_best_cw.keras",
                                    monitor="val_accuracy", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
]
cnn.fit(train_ds, validation_data=val_ds, epochs=30,
        callbacks=cb, class_weight=class_weight)
```

| Callback | Role |
|---|---|
| `ModelCheckpoint` | Saves the model every time `val_accuracy` reaches a new best, so the best version is always on Drive even if the session disconnects. |
| `EarlyStopping` | Stops training when `val_accuracy` has not improved for 5 consecutive epochs, and restores the weights of the best epoch. |
| `ReduceLROnPlateau` | Halves the learning rate when `val_loss` stops improving for 2 epochs. |

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
- **Measuring success:** not overall accuracy alone, but **macro F1** and **per-class recall**.
- **Effect observed:** even the rarest classes are learned well. For example `Apple___Cedar_apple_rust` (41 test images) reaches F1 = 1.000, `Potato___healthy` (23 test images) reaches 0.933, and macro F1 (0.945) is almost equal to overall accuracy (0.950), which means the small classes are not being sacrificed for the large ones.
- **Side effect:** class weighting makes the model more willing to predict small classes. Some of them reach recall 1.000 with lower precision (for example `Raspberry___healthy` and `Corn_(maize)___Cercospora_leaf_spot`, both precision 0.786), meaning a few images from other classes get assigned to them.
- **Fallback:** if some small classes stay weak, try oversampling (repeating small-class images with augmentation) **instead of** class_weight, not together with it, and compare by macro F1 on validation.
- SMOTE was not used because it is designed for tabular data, and interpolating pixels between two images produces an unrealistic image.
- Note: no run without class weights was done for a direct comparison, so the benefit is supported by the per-class results above rather than by an ablation.

#### Training results and Early Stopping

Training ran for **20 epochs** out of the 30 allowed, and then stopped by itself:

| Epoch | Train accuracy | Val accuracy | Learning rate |
|---|---|---|---|
| 1 | 0.505 | 0.606 | 1e-3 |
| 5 | 0.899 | 0.756 | 5e-4 |
| 10 | 0.936 | 0.578 | 5e-4 |
| 11 | 0.953 | 0.914 | 2.5e-4 |
| **15 (best)** | 0.969 | **0.949** | 1.25e-4 |
| 18 | 0.974 | 0.943 | 6.25e-5 |
| 20 | 0.977 | 0.939 | 3.125e-5 |

How Early Stopping behaved:
- The best `val_accuracy` was **0.9493 at epoch 15**.
- During epochs 16 to 20 (five epochs in a row) it never exceeded that value (0.935, 0.909, 0.943, 0.902, 0.939), so `patience=5` triggered and training stopped at epoch 20. Re-running the training gave the same behavior.
- `restore_best_weights=True` returned the model to the epoch-15 weights, and `ModelCheckpoint` had already saved the same best version to Drive.
- This saved about 10 epochs (~20 minutes of GPU time) and avoids overfitting to the training set (train accuracy kept rising to 0.977 while validation did not improve).

Other observations:
- Validation accuracy was very unstable in the first 10 epochs (between 0.35 and 0.88) while the learning rate was still high and BatchNorm statistics were settling. It became stable once `ReduceLROnPlateau` lowered the learning rate to 2.5e-4 and below.
- The learning rate was reduced automatically five times, from 1e-3 down to 3.125e-5.

### Stage 6: Evaluation

Decisions and comparisons were made on the **validation** set, and the **test** set was used **once** at the end on the saved best model.

| Metric | Validation | Test |
|---|---|---|
| Images | 8,143 | 8,143 |
| Accuracy | 0.949 | **0.950** |
| Macro F1 | 0.943 | **0.945** |
| Weighted F1 | 0.949 | 0.950 |
| Macro precision | 0.939 | 0.943 |
| Macro recall | 0.952 | 0.953 |
| Loss | – | 0.155 |

**Reading the results:**
- Validation and test results are almost identical, so the model **generalizes to unseen images** and was not overfit to the validation set through checkpoint and early-stopping selection.
- Macro recall (0.953) is slightly higher than macro precision (0.943), which matches the effect of class weights described above.
- The same classes are weak in both validation and test, so this is a real weakness of the model and not noise.

**Best classes (test F1):** `Apple___Cedar_apple_rust`, `Corn_(maize)___Common_rust_`, and `Corn_(maize)___healthy` all reach 1.000. `Strawberry___Leaf_scorch` (0.997) and `Blueberry___healthy` (0.987) are also very strong.

**Weakest classes (test F1):**

| Class | Precision | Recall | F1 | Comment |
|---|---|---|---|---|
| `Tomato___Spider_mites Two-spotted_spider_mite` | 0.917 | 0.789 | 0.848 | Low recall: often confused with other tomato diseases |
| `Tomato___Target_Spot` | 0.809 | 0.910 | 0.857 | Low precision: other tomato classes get assigned to it |
| `Tomato___healthy` | 0.785 | 1.000 | 0.880 | Diseased tomato leaves sometimes predicted as healthy |
| `Raspberry___healthy` | 0.786 | 1.000 | 0.880 | Small class, over-predicted |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 0.786 | 1.000 | 0.880 | Small class, over-predicted |
| `Pepper,_bell___Bacterial_spot` | 0.992 | 0.793 | 0.881 | Low recall |
| `Apple___Black_rot` | 0.809 | 1.000 | 0.894 | Other apple classes get assigned to it |
| `Pepper,_bell___healthy` | 0.979 | 0.824 | 0.895 | Low recall |

Most errors happen between **visually similar tomato diseases** and between a healthy class and a disease class of the same plant.

**Important limitation:** PlantVillage images are taken in the lab on a uniform background. The 95% above is measured on the same kind of images, so on real field photos or images from the internet the accuracy will likely be lower (domain shift). In addition, a softmax classifier always assigns an image to one of the 38 classes, even when the image is not a leaf at all.

### Stage 7: Using the model

- Input: an RGB image resized to **128×128**, with pixel values left as **0–255** (the model divides by 255 itself).
- Output: 38 probabilities, ordered by `sorted(train_df["label"].unique())`. The same class list must be saved with the model (`classes.json`).
- The best model is saved as `cnn_best_cw.keras`.

## 6. Roadmap

- [x] Download the dataset and inspect its structure and class distribution
- [x] Build the image table and check for corrupted files
- [x] Detect and remove duplicate images (21 images)
- [x] Split the dataset (train / val / test)
- [x] Build the image loading pipeline
- [x] Define the CNN architecture
- [x] Train the model with `class_weight`, `EarlyStopping`, and `ReduceLROnPlateau`
- [x] Evaluate on validation, then on test (95.0% accuracy, 0.945 macro F1)
- [ ] Error analysis (confusion matrix, focusing on tomato diseases)
- [ ] Test on images from outside the dataset and add an `Other` class (plus a confidence threshold) so non-leaf images are rejected
- [ ] Try transfer learning (e.g. EfficientNetB0 or MobileNetV2) and higher input resolution
- [ ] Build the web interface for uploading leaf images

## 7. Getting Started

The project is written for **Google Colab**:

1. Open the notebook in Colab and enable a GPU (Runtime → Change runtime type).
2. Create a Kaggle API token and store it in **Colab Secrets** as `KAGGLE_API_TOKEN` (never hardcode it in the notebook or push it to GitHub).
3. Run the cells in order (do not use Run all): download and load, then build the table, clean, and split (once only), then model and training.
4. In any later session, read `train.csv`, `val.csv`, and `test.csv` from Drive instead of splitting again.

```bash
pip install kaggle pillow pandas scikit-learn matplotlib tqdm tensorflow
```
