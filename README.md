# ML OPS LAB 4 TensorBoard

## Overview
This project trains a neural network classifier to predict Spotify song popularity (Low/Medium/High) from audio features. It runs three experiments and compares them in TensorBoard:

- Baseline model with Adam optimizer
- Dropout model with Adam optimizer
- Baseline model with SGD optimizer

The workflow includes data cleaning, train/test split, feature scaling, model training, TensorBoard visualization, and evaluation metrics.

## Dataset
The notebook expects a CSV file named `dataset.csv` with these columns:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration_ms
- popularity

`popularity` is bucketed into three classes:

- 0 = Low (0-33)
- 1 = Medium (34-66)
- 2 = High (67-100)

## Requirements
- Python 3.9+ recommended
- Packages listed in `requirements.txt`

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Notebook
Open the notebook and run all cells:

- Notebook: `spotify_tensorboard_lab (1).ipynb`
- Update this line for local runs:
  - `dataset_path = 'dataset.csv'`

TensorBoard options:

- In-notebook: `%tensorboard --logdir logs/spotify`
- Terminal: `tensorboard --logdir logs/spotify`

Training logs are written to `logs/spotify` (ignored by Git).

## Outputs
- Accuracy and loss curves
- Classification report
- Confusion matrix
- TensorBoard runs comparing experiments

## Project Structure
```
.
├── dataset.csv
├── spotify_tensorboard_lab (1).ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Notes
- The dataset is imbalanced, which can cause weak performance on the High class.
- Suggested improvements include class weighting, oversampling, or adjusting the bucket thresholds.
