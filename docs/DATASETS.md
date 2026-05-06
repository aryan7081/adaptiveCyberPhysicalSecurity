# Local datasets (not in Git)

Large CIC-IDS CSV files are listed in `.gitignore` to keep the repository lightweight.

- Place merged flows at `data/raw/cicids2017_all.csv` (see `config/config.yaml`).
- Original segment exports may live under `archive/` locally.

Regenerate the merged file by concatenating the segment CSVs from your archive, or adjust `dataset.train_file` to point at a single segment for smaller experiments.
