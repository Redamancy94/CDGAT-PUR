# CDGAT-PUR

CDGAT-PUR is a multi-omics disease-gene prioritization method that combines
ANM-informed dynamic graph attention with positive-unlabeled ranking.

## Repository layout

```text
CDGAT-PUR/
├── run_cdgat_pur.py
├── cdgat_pur/
│   ├── data.py
│   ├── model.py
│   └── training.py
├── data/
│   ├── cgga_glioma/
│   ├── graphs/
│   └── splits/
├── requirements.txt
└── README.md
```

## Environment

Python 3.10 and an NVIDIA CUDA GPU are recommended.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Select the PyTorch index URL compatible with the local CUDA version.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Reproduce the CGGA-GLIOMA experiment

```bash
python run_cdgat_pur.py \
  --device cuda \
  --seeds 1 2 3 4 5 \
  --max-epochs 2000 \
  --patience 100
```

## Fixed training configuration

| Parameter                   | Value                             |
| --------------------------- | ---------------------------------:|
| Omics                       | expression, methylation, mutation |
| First-layer attention heads | 8                                 |
| Hidden features per head    | 16                                |
| Dropout                     | 0.2                               |
| Learning rate               | 0.005                             |
| Weight decay                | 0.0005                            |
| Maximum epochs              | 2000                              |
| Early-stopping patience     | 100                               |
| Within-seed folds           | 5                                 |
| Random seeds                | 1, 2, 3, 4, 5                     |
