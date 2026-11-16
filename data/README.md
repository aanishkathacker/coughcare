# Data directory

Large COUGHVID files are **not committed** to this repository.

```
data/raw/public_dataset.zip          # 951 MB, MD5 5c30a8b00c8bb7783a2c15a48cb8ea9e
data/raw/public_dataset/             # extracted audio + metadata_compiled.csv
data/processed/train_manifest.csv    # deterministic 1,000-clip balanced subset
data/processed/selection_summary.json
data/processed/features.npz          # cached 395-dimensional feature matrix
```

Recreate everything with:

```bash
python scripts/download_data.py     # download + MD5 + zip integrity + extract
python scripts/prepare_data.py      # build the balanced manifest
python scripts/train_model.py       # features + 5-fold CV + model + metrics
```

The subset is selected with `RANDOM_STATE=42`, so the manifest is reproducible.
