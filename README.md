# Pneumonia Detection from Chest X-rays (Deep Learning)

Binary classification of chest X-ray images (**NORMAL** vs **PNEUMONIA**) using a baseline CNN and a fine-tuned MobileNetV2 (transfer learning), built in TensorFlow/Keras and designed to run in Google Colab.

> ## Medical disclaimer
> This is an **educational research project**. It is **not** a validated medical diagnostic tool, has not been clinically evaluated, and must not be used to diagnose, treat, or rule out pneumonia or any other condition. Always consult a qualified healthcare professional.

## Project status

| Item | Status |
|---|---|
| Data loading, split, leakage checks, class weighting | Done (run in Colab) |
| Baseline CNN, MobileNetV2 (frozen), MobileNetV2 (fine-tuned) | Done (run in Colab) |
| Evaluation: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix | Done (run in Colab) |
| Threshold tuning, bootstrap confidence intervals, Grad-CAM | Done (run in Colab) |
| Standalone `src/` scripts, command-line training/prediction | **Not implemented** |
| Patient-level split, external validation | **Not implemented** |

**Reproducibility note:** the notebook in this repo is a cleaned and re-ordered version of the original Colab notebook (see [Changes from the original notebook](#changes-from-the-original-notebook)). The metrics below come from the **original run**. The cleaned notebook has been syntax-checked but **has not been re-run end-to-end**, so numbers may differ slightly when you run it (random initialisation, GPU non-determinism).

## 1. Problem statement and objectives
Pneumonia is commonly assessed with chest radiographs. This project explores whether a deep learning model can separate pneumonia from normal chest X-rays, and examines how reliable the result is.

Objectives:
- Build a clean train / validation / test pipeline without data leakage.
- Compare a simple CNN against transfer learning with MobileNetV2.
- Report metrics that matter for a screening-style task (sensitivity **and** specificity, not only accuracy).
- Be honest about limitations.

## 2. Dataset
- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle (`paultimothymooney/chest-xray-pneumonia`), derived from Kermany et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, Cell, 2018.
- **Licence / attribution:** check the dataset page for the current licence and citation requirements before redistributing anything. The dataset is **not** included in this repository.
- **Contents (as loaded in the run):**

| Official split | NORMAL | PNEUMONIA |
|---|---:|---:|
| train | 1341 | 3875 |
| val (16 images, **not used**) | 8 | 8 |
| test | 234 | 390 |

- **Splits actually used:** the official `train` folder is split 80/20 (stratified, seed 42) into train and validation; the official `test` folder is untouched.

| Split | Images |
|---|---:|
| Train | 4172 |
| Validation | 1039 (5 exact duplicates of training images removed) |
| Test | 624 |

## 3. Technologies
Python, TensorFlow / Keras, `kagglehub`, NumPy, pandas, scikit-learn, Matplotlib, seaborn, Pillow. Developed on Google Colab (GPU runtime).

## 4. Workflow
1. Download the dataset with `kagglehub`.
2. Inspect images (sizes, modes, unreadable files) and class balance.
3. Split train into train/validation; check for path and exact-file (MD5) leakage; compute class weights from training labels only.
4. Build a `tf.data` pipeline (RGB, resize to 224×224).
5. Train the baseline CNN and the frozen MobileNetV2; select by **validation AUC**.
6. Evaluate on the untouched test set; bootstrap confidence intervals; error analysis.
7. Tune the decision threshold on validation data.
8. Fine-tune the top 30 MobileNetV2 layers; re-evaluate.
9. Grad-CAM visualisation, save model, predict on a new image.

## 5. Model architecture
- **Baseline CNN:** augmentation → rescale → 4 × (Conv3×3 + MaxPool) with 16/32/64/128 filters → global average pooling → dropout 0.4 → sigmoid.
- **MobileNetV2:** ImageNet-pretrained backbone → global average pooling → dropout 0.3 → sigmoid. Stage 1: backbone frozen (Adam 1e-3, 10 epochs max). Stage 2: top 30 layers unfrozen with BatchNorm frozen (Adam 1e-5, 8 epochs max).
- **Training details:** binary cross-entropy, class weights (NORMAL 1.944, PNEUMONIA 0.673), in-model augmentation (small rotation/zoom/translation/contrast), early stopping and checkpointing on validation AUC, learning-rate reduction on plateau.

## 6. Setup

### Google Colab (recommended)
1. Upload or open `notebooks/pneumonia_detection.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **GPU**.
3. Runtime → Run all. The dataset downloads automatically through `kagglehub`. If Kaggle asks for credentials, follow the on-screen prompt. **Never commit `kaggle.json` or any API key.**

### Local (untested)
```bash
git clone https://github.com/<your-username>/pneumonia-detection.git
cd pneumonia-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/pneumonia_detection.ipynb
```
The notebook was developed for Colab; local use has not been tested and a GPU is strongly recommended.

## 7. Train and evaluate
Run the notebook top to bottom. Training is in Sections 4 and 7, evaluation in Sections 5 and 7, threshold tuning in Section 6. There is no separate `train.py` / `evaluate.py`.

## 8. Predict on a sample X-ray
Run the whole notebook (so the trained model exists), then run Section 10 ("Predict on your own X-ray") and upload a JPG/PNG. It prints a label and a score with an educational-use warning. Weights are not stored in this repo; see [`models/README.md`](models/README.md).

## 9. Results
All numbers are from the original Colab run on the **624-image official test set**, threshold 0.5 unless stated. Single run, no repeated seeds.

| Model | Accuracy | Precision | Recall (sensitivity) | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.829 | 0.894 | 0.823 | 0.838 | 0.857 | 0.910 |
| MobileNetV2 (frozen) | 0.853 | 0.818 | 0.982 | 0.637 | 0.893 | 0.963 |
| MobileNetV2 (fine-tuned) | 0.877 | 0.842 | 0.987 | 0.692 | 0.909 | 0.973 |

Confusion-matrix counts (TN / FP / FN / TP): baseline 196 / 38 / 69 / 321; frozen 149 / 85 / 7 / 383; fine-tuned 162 / 72 / 5 / 385.

Other results from the run:
- Best **validation** AUC: baseline CNN 0.964, MobileNetV2 frozen 0.988, MobileNetV2 fine-tuned 0.994.
- Bootstrap 95% CI for frozen MobileNetV2 on the test set: accuracy 0.827–0.881, ROC-AUC 0.949–0.976. (Not computed for the fine-tuned model.)
- Threshold tuned on validation (0.349) for frozen MobileNetV2: test accuracy 0.83, recall 0.99, specificity 0.564 (TN 132 / FP 102 / FN 4 / TP 386). This catches more pneumonia cases but raises false alarms.

Figures (baseline CNN and **frozen** MobileNetV2 only; the fine-tuned model was added later in the run and is not in these plots):

![Training curves](results/training_curves.png)
![Confusion matrices and ROC curves](results/confusion_matrices_and_roc.png)

Grad-CAM overlays are produced in Section 8 of the notebook (qualitative only, not shown here to avoid redistributing dataset images).

### How to read these results
- The models are **very sensitive but not very specific**: they catch nearly all pneumonia cases in this test set but flag roughly 30–36% of normal images as pneumonia.
- Validation accuracy (about 95–96%) is much higher than test accuracy (85–88%). Possible reasons, **not verified**: the validation split is random at image level and the dataset contains several images per patient (optimistic validation), and the official test set may differ in distribution from the training set.
- The test set was evaluated several times during development (baseline, frozen, tuned threshold, fine-tuned), so the reported test numbers are slightly optimistic. Model selection itself used validation AUC only.

## 10. Limitations and future work
Limitations:
- One public dataset, one run, no external validation; results will not necessarily transfer to other hospitals, scanners, or populations.
- Image-level split; patient overlap between train and validation is possible.
- Low specificity (64–69% at threshold 0.5).
- Pneumonia labels come from the dataset, not from this project; label noise is possible.
- Grad-CAM was inspected qualitatively only.
- The cleaned notebook has not been re-run end-to-end (see Project status).

Future work:
- Patient-level (grouped) train/validation split, and repeated runs with different seeds.
- Calibration and threshold analysis on an independent set.
- Evaluate the fine-tuned model with confidence intervals.
- Refactor into tested `src/` modules and add a command-line `predict.py`.
- Try other backbones (EfficientNet, DenseNet) and external datasets.

## Changes from the original notebook
- **Cells re-ordered:** the original trained models in a cell that ran before `train_ds`, `val_ds` and `class_weight` were defined.
- **Removed broken/duplicate cells:** a stray character causing a syntax error in the training-curve cell, two Grad-CAM cells that failed (one `ValueError`, one `SyntaxError`), and an indentation-broken duplicate of the leakage check. The working Grad-CAM version and the working leakage check were kept.
- **Saved model fixed:** the original saved the pre-fine-tuning MobileNetV2 even though the fine-tuned model had the higher validation AUC. The cleaned notebook saves the fine-tuned model.
- **Minor:** Colab-only code is guarded so the notebook does not crash outside Colab; a file-handle leak in the MD5 helper was fixed; the unused official `val` folder is documented.

## Repository structure
```
pneumonia-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/pneumonia_detection.ipynb
├── results/
│   ├── training_curves.png
│   └── confusion_matrices_and_roc.png
└── models/README.md        # how to obtain/generate weights (weights not committed)
```

## Acknowledgements
Dataset: Kermany, D. S. et al. (2018), *Cell*; Kaggle upload by Paul Mooney. Pretrained backbone: MobileNetV2 (Sandler et al., 2018) with ImageNet weights via Keras.
