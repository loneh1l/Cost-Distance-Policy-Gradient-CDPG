# CoDi-PG (CDPG)

## Dataset

The active dataset loader is GBA-CDNet. The dataset is expected under
`data/GBA-CDNet/` with subdirectories `GBA-CDNet-1` through `GBA-CDNet-9`.
Each subgraph directory contains:

- `nodes.csv`
- `edges.csv`

The `rand` and `tn` dataset loading paths have been removed from
`utils/data_loader.py`. Their command-line options still exist in
`args_config.py`, but `main.py` only has a dedicated loader for `gba`; other
dataset choices fall through to the generic `DataLoader` path.

## Methods

The runnable methods in `main.py` are:

- `cdpg`
- `held`
- `chekuri`
- `meyerson`
- `gurobi`

Legacy parser entries for removed methods may still appear in `args_config.py`,
but the corresponding method files and `main.py` execution branches have been
removed.

## Scripts

Single-run scripts are in `scripts/`:

- `scripts/cdpg.sh`: CDPG on GBA subgraph 3.
- `scripts/held.sh`: Held baseline on GBA subgraph 3.
- `scripts/chekuri.sh`: Chekuri baseline on GBA subgraph 3.
- `scripts/meyerson.sh`: Meyerson baseline on GBA subgraph 3.
- `scripts/gurobi.sh`: Gurobi baseline on GBA subgraph 3 for Linux/WSL.

Run shell scripts from Linux:

```bash
chmod +x scripts/*.sh
bash scripts/cdpg.sh
bash scripts/held.sh
bash scripts/chekuri.sh
bash scripts/meyerson.sh
bash scripts/gurobi.sh
```

## Notes

All current scripts use `dataset=gba` and default to `subgraph=3`. Outputs are
written under `logs/` unless the script-specific log directory environment
variable is overridden.
