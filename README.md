# Fluorescent Protein Emission Predictor

Predicts the emission wavelength (in nanometers) of fluorescent proteins from their amino acid sequence, then uses evolutionary search to discover novel protein sequences with target emission properties. Uses ESM-2 protein language model embeddings with Ridge regression and MLP models trained on 839 proteins from FPBase.

**Full pipeline**: Train emission predictor → Evolutionary search for novel sequences → Score candidates with ESM3

**Best performance**: Ensemble model achieves ~21 nm mean absolute error on emission prediction.

## Setup

1. Install Python 3.10+ (recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html)):
   ```bash
   conda create -n proteins python=3.12
   conda activate proteins
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For ESM-C 600M support:
   ```bash
   pip install esm
   ```

## Quick start

Run the full pipeline (download data, generate embeddings, train, evaluate):

```bash
python pipeline.py
```

This takes ~10 minutes the first time (downloading model + embedding 839 sequences). Subsequent runs take seconds because everything is cached.

## Predicting on your own sequences

After training, predict emission wavelength for any amino acid sequence:

```bash
python predict.py artifacts/esm2/ensemble.json --sequence "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL"
```

For multiple sequences from a FASTA file:

```bash
python predict.py artifacts/esm2/ensemble.json --fasta candidates.fasta
```

As a Python library (for search algorithms):

```python
from predict import EmissionPredictor

predictor = EmissionPredictor("artifacts/esm2/ensemble.json")
wavelength = predictor.predict("MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL")
# -> 509.3

wavelengths = predictor.predict_batch(["MVSK...", "MEEL...", ...])
# -> np.array([509.3, 527.1, ...])
```

## Evolutionary search

Discovers novel fluorescent protein sequences predicted to emit at a target wavelength. Uses ESM-C as a mutation prior and the trained ensemble as a fitness oracle.

```bash
# Search for green FP (~520 nm)
python search.py --target 520

# Customize search parameters
python search.py --target 620 --population-size 80 --max-generations 50 --output results/red/
```

### How it works

1. Seeds the population with natural fluorescent proteins ancestrally related to the target wavelength
2. Each generation: mutates parents using ESM-C language model probabilities, filters by pseudo-log-likelihood, scores fitness
3. Fitness combines emission accuracy, plausibility (PLL), novelty, and diversity
4. Uses k-means clustering to preserve sequence diversity across generations
5. Converges when enough candidates hit the target emission within tolerance

### Configuration

All search hyperparameters are in `search_config.py`. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target` | (required) | Target emission wavelength in nm |
| `--population-size` | 50 | Population size per generation |
| `--num-seeds` | 8 | Number of seed proteins |
| `--max-generations` | 100 | Maximum generations |
| `--mutations-per-parent` | 20 | Mutations per parent per generation |
| `--output` | `results/` | Output directory |
| `--artifact` | `artifacts/cross/ensemble.json` | Path to trained ensemble |

### Output

Results are saved to the output directory:

- `results.json` — Top 20 candidates with sequences, mutations, predicted emission, fitness scores
- `generations.json` — Per-generation metrics (best fitness, emission, timing)
- `top_candidates.fasta` — FASTA-formatted sequences for downstream tools

## Candidate scoring

After search, score candidates for fluorescence likelihood using ESM3 structural predictions. Evaluates on 4 axes:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Chromophore** | 0.35 | Conservation of GFP chromophore-critical residues (G67, aromatic66, R96, E222) |
| **Secondary structure** | 0.25 | Beta-barrel fold composition via ESM3 SS prediction |
| **pTM** | 0.25 | Fold confidence from ESM3 structure prediction (pTM/pLDDT) |
| **Homology** | 0.15 | Sequence identity to known fluorescent proteins in FPBase |

### Setup

Scoring requires the ESM3 model from HuggingFace (gated access):

```bash
# 1. Accept the license at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
# 2. Log in to HuggingFace
huggingface-cli login

# 3. Verify auth + download model
python scripts/score_candidates.py --setup
```

### Usage

```bash
# Score candidates from a search run (pass the results folder)
python scripts/score_candidates.py --input results/full_test/

# Or pass results.json directly
python scripts/score_candidates.py --input results/full_test/results.json

# Run self-test (EGFP positive control + insulin negative control)
python scripts/score_candidates.py --input results/full_test/ --self-test

# Show only top 5
python scripts/score_candidates.py --input results/full_test/ --top-n 5
```

### Output

Scoring writes to the input directory:

- `scores.json` — Full scored results with per-component breakdowns
- `scores_report.txt` — Human-readable ranked report with explanations
- `top10.json` — Top 10 candidates summary

## Trying different models

| Command | What it does |
|---------|-------------|
| `python pipeline.py` | Train ensemble (Ridge + MLP) with ESM-2 embeddings |
| `python pipeline.py run --model ridge` | Ridge regression only |
| `python pipeline.py run --model mlp` | MLP only |
| `python pipeline.py run --embedding esmc600m` | Use ESM-C 600M instead of ESM-2 |
| `python pipeline.py train --model ridge --pca 128` | Ridge with custom PCA dimensions |

Individual steps:
```bash
python pipeline.py download           # just download FPBase data
python pipeline.py embed              # just generate embeddings
python pipeline.py train              # just train models
python pipeline.py evaluate           # just evaluate saved models
```

## What the output means

- **MAE** (Mean Absolute Error): Average prediction error in nanometers. Lower is better. Our ensemble achieves ~21 nm, meaning predictions are off by ~21 nm on average.
- **RMSE** (Root Mean Squared Error): Like MAE but penalizes large errors more. Useful for spotting outliers.
- **R2** (R-squared): How much variance the model explains. 1.0 = perfect, 0.0 = no better than predicting the mean. Our models achieve ~0.85.

Fluorescent protein emission wavelengths range from ~380 nm (UV) to ~1000 nm (near-IR), so a 21 nm error is quite good for most applications.

## Project structure

```
pipeline.py                    CLI entry point for training
predict.py                     Inference script + importable library
data.py                        FPBase download + caching + train/test split
embeddings.py                  ESM-2 and ESM-C embedding generation
models.py                      Model definitions, training, saving, loading
search.py                      Evolutionary search for novel FP sequences
search_config.py               Search hyperparameters (dataclass)
scripts/score_candidates.py    ESM3-based candidate scoring pipeline
requirements.txt               Dependencies

cache/                         Auto-created, cached intermediate data
artifacts/                     Saved trained models
results/                       Search output (results.json, generations.json, etc.)
```

## Troubleshooting

**Out of memory during embedding**: ESM-2 (650M parameters) needs ~3 GB RAM. Close other applications or use a machine with more memory.

**MPS (Apple Silicon) errors**: If you see MPS-related errors, set `PYTORCH_ENABLE_MPS_FALLBACK=1` or force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python pipeline.py
```

**Missing `esm` package**: ESM-C support requires `pip install esm`. ESM-2 works without it (uses HuggingFace transformers).

**Want to re-embed**: Delete the relevant `cache/esm2/` or `cache/esmc600m/` directory and re-run.

**Want to retrain**: Use the `--force` flag: `python pipeline.py train --force`

**ESM3 scoring auth errors**: Make sure you've accepted the license at HuggingFace and logged in with `huggingface-cli login`. Run `python scripts/score_candidates.py --setup` to verify.
