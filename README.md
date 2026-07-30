# ESM-ECForest

ESM-ECForest is a two-stage framework that combines ESM-2 protein language model embeddings with Random Forest classifiers for hierarchical enzyme annotation.

## Overview

The prediction workflow consists of two stages:

1. **Enzyme/non-enzyme classification**
2. **Hierarchical Enzyme Commission (EC) number prediction** for sequences classified as enzymes

ESM-ECForest provides:

- Enzyme/non-enzyme classification
- Hierarchical EC number prediction
- ESM-2 protein sequence embeddings
- Random Forest classifiers

## Installation

Clone the repository:

```bash
git clone https://github.com/Xiao88866/ESM-ECForest
cd ESM-ECForest
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate enzyllm_env
```

The environment uses Python 3.9 and includes PyTorch, Transformers, ESM, scikit-learn, pandas, NumPy, and joblib.

## Pre-trained models

Due to GitHub file size limits, the pre-trained model files are hosted on Zenodo:

https://doi.org/10.5281/zenodo.21699471

Download the following files:

- `ec_classifier.joblib` 
- `enzyme_classifier.joblib`

Place both files in the models/ directory.

## Usage

Input protein sequences should be provided in FASTA format. 

Run prediction on a FASTA file:

```bash
python multi_test_batch_predict.py input.fasta
```

Prediction results are saved as `input_predictions.csv` in the current working directory.

## Citation

If you use the pre-trained models, please cite the Zenodo record:

https://doi.org/10.5281/zenodo.21699471

Citation information for the associated manuscript will be added after publication.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
