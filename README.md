# MLP Face Recognition

A face recognition pipeline using classical CV + ML: **PCA (eigenfaces)** for dimensionality reduction, **LDA (fisherfaces)** for class separation, and an **MLP classifier** for identification.

## Pipeline

1. **Preprocess** — Images from `dataset/faces/<person_name>/` are grayscaled, resized to 300×300, and flattened; folder names become class labels.
2. **Split** — 75% train / 25% test.
3. **PCA** — Top 150 components reduce dimensionality (visualized as eigenfaces).
4. **LDA** — Projects PCA features to maximize separability between people.
5. **MLP** — A two-hidden-layer (10, 10) `MLPClassifier` trains on the LDA features.
6. **Evaluate** — Predicts on the test set, reporting accuracy and a gallery of predicted vs. true labels.

## Project structure

```
MLP-Face-Recognition/
├── MLP Face Recognition.ipynb   # Main notebook: full pipeline
├── dataset/
│   ├── faces/                   # Face images, one subfolder per person
│   │   ├── <Person Name>/
│   │   │   ├── face_1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── Iris/                    # Iris dataset (iris.data, iris.names)
└── README.md
```

## Requirements

```bash
pip install numpy matplotlib scikit-learn opencv-python
```

## Usage

1. Add face images under `dataset/faces/<person_name>/` — one folder per person.
2. Run the notebook top to bottom:

```bash
jupyter notebook "MLP Face Recognition.ipynb"
```

It prints dataset stats, shows the top eigenfaces, trains the MLP, then reports test accuracy with a gallery of predicted vs. true labels.

## Notes

- Included dataset is a small custom set, not a public benchmark.
- The notebook has commented-out code for loading scikit-learn's `fetch_lfw_people` (LFW) as an alternative data source.
- PCA components, MLP hidden-layer sizes, and split ratio are configured directly in the notebook.
