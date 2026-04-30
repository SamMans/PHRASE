```markdown
# Benchmarking Framework for Gait Phase Recognition

This directory contains a complete framework for training, evaluating, and statistically comparing multiple machine learning models for real-time gait phase recognition from wearable IMU sensors. The framework implements state-of-the-art deep learning architectures alongside a novel heuristic-Bayesian inference system (PHRASE).

## Overview

The framework supports:

- **5 deep learning architectures**: LSTM, CNN-LSTM, ConvGRU, ST-GCN, and Transformer
- **End-to-end pipeline**: Training → Inference → Statistical analysis
- **Real-time simulation**: Sequential window-by-window processing to mimic real-world deployment
- **Subject-level cross-validation**: K-fold evaluation with unseen subjects
- **Comprehensive metrics**: Accuracy, precision, recall, F1, specificity, inference time, transition delay
- **Statistical analysis**: Paired Wilcoxon tests with effect sizes and confidence intervals

## Repository Structure

```
benchmarking/
├── train.py                    # Main training script
|
├── inf.py                      # Main inference script
|
├── analysis.py                 # Statistical analysis script
|
├── util/
|
│   ├── benchmarkTrain.py       # Training class (BMtrainer)
|   |
│   └── benchmarkInf.py         # Inference classes (BMinf + model wrappers)
|
└── README.md                   # This file
```

## File Descriptions

### `train.py`
Command-line interface for training models. Parses user arguments and delegates to `BMtrainer`.

**Usage:**

```bash
python3 train.py --model "add type of training model here: lstm, cnn-lstm, convGRU, gnn, transformer" --path /path/to/dataset --model lstm --ratio "desired ratio: training freq. / test freq."
```

note 1: remove "..." and add your desired option
note 2: training freq. is the sensory sampling frequency of the training set's IMUs
note 3: test fre. is the expected sensory sampling frequency of the test set
note 4: the ratio is not required for PHRASE's MATLAB training code because PHRASE adapts naturally to any frequency

### `util/benchmarkTrain.py`
Contains the `BMtrainer` class responsible for:
- Loading and preprocessing CSV datasets
- Segmenting continuous gait bouts
- Extracting sliding window features
- Building and training specified model architectures
- Saving model weights and metadata
- Generating learning curves

**Supported models:** `lstm`, `cnn-lstm`, `convGRU`, `gnn`, `transformer`

### `inf.py`
Command-line interface for inference. Evaluates trained models on test subjects.

**Usage:**
```bash
python3 inf.py --dataset "select your dataset from the resources file e.g., BLISS_inference" --pretrained /path/to/checkpoints --model "select desired model e.g., lstm"
```

note : remove "..." and add your desired option

### `util/benchmarkInf.py`
Contains inference classes:
- `BMinf`: Main inference orchestrator
- `phrase_inf`: Heuristic-Bayesian inference (PHRASE)
- `lstm_inf`, `cnn_lstm_inf`, `convGRU_inf`, `gnn_inf`, `transformer_inf`: Deep learning model wrappers

All inference classes maintain sequential state for real-time processing.

### `analysis.py`
Statistical post-processing script. Performs paired Wilcoxon tests comparing PHRASE against all benchmark models.

**Usage:**
```bash
python3 analysis.py --ds1 dataset1 --ds2 dataset2 --metric "choose desired metric e.g., accuracy" --phase "choose desired gait phase e.g., LR"
```

note 1: you can add up to 6 datasets for analysis --ds1 --ds2 ... --ds6
note 2: remove "..." and add your desired option

## Data Format Requirements

### Input CSV Files
Each CSV file must contain:
- **Required columns**: `Right_Shank_Ax`, `Right_Shank_Az`, `Right_Shank_Gy`, `Left_Shank_Ax`, `Left_Shank_Az`, `Left_Shank_Gy`, `Mode`, `phase`
- **Naming convention**: `ABXXX_circuitYY.csv` (e.g., `AB001_circuit01.csv`)
- **`Mode` column**: `1` indicates walking activity
- **`phase` column**: Gait phase labels (see metadata for code mapping)

### Metadata File (`metadata.json`)
Located in the dataset directory. Must contain:
```json
{
    "signal_parameters": {
        "sampling_frequency": 250
    },
    "gait_phases": {
        "phase_names": ["LR", "MST", "TS", "PSW", "SW"]
    }
}
```

## Model Architectures

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **LSTM** | Standard LSTM sequence classifier | Tx: sequence length, n_a: 64 activations |
| **CNN-LSTM** | CNN feature extractor + LSTM sequence model | 10 Conv2D filters, 3x3 kernel |
| **ConvGRU** | Convolutional GRU following Shi et al. | n_a: 32 activations, Conv2D gates |
| **ST-GCN** | Spatial-Temporal Graph Convolutional Network | Predefined adjacency for 6 IMU sensors |
| **Transformer** | Vanilla Transformer encoder | d_model: 64, 4 attention heads, 2 layers |
| **PHRASE** | Heuristic-Bayesian hybrid system | Prior distributions + ANN + heuristic rules |

## Training Workflow

1. **Data organization**: CSV files are split into seen/unseen subjects
2. **Gait bout segmentation**: Continuous walking segments are extracted
3. **Feature extraction**: Sliding windows with overlap
4. **Sequence formation**: Windows grouped into sequences (length `s`)
5. **Model training**: 10 epochs with Adadelta (LSTM/CNN-LSTM) or Adam (others)
6. **K-fold cross-validation**: Each subject serves as unseen test subject once
7. **Model saving**: `.keras` weights + `.json` metadata

## Inference Workflow

1. **Real-time simulation**: Windows are processed sequentially
2. **Sequence buffering**: Model receives full sequence after `s*w` samples
3. **Transition detection**: Phase changes are detected and timed
4. **Cool-down period**: Predictions near true transitions are excluded for fair evaluation
5. **Metrics computation**: Accuracy, precision, recall, F1, specificity, inference time, transition delay

## Statistical Analysis

The `analysis.py` script computes:

- **Paired Wilcoxon signed-rank test** (non-parametric, suitable for non-normal paired differences)
- **Rank-biserial correlation (RBC)** as effect size
- **Common Language Effect Size (CLES)**
- **Bootstrap 95% confidence intervals** for median differences

## Example Usage

### 1. Train a model
```bash
cd benchmarking
python3 train.py --path ../resources/BLISS_training --model lstm --ratio 1
```

### 2. Run inference on test subjects
```bash
python3 inf.py --dataset BLISS_inference --pretrained ../checkpoints --model lstm --eval test
```

### 3. Compare all models statistically
```bash
python3 analysis.py --ds1 BLISS_inference --ds2 BLISS_inference_severe --metric accuracy
```

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `*_model_W_*.keras` | `../checkpoints/` | Trained model weights |
| `*_model_W_*.json` | `../checkpoints/` | Model metadata (window size, sequence length, modalities) |
| `*_results.json` | Dataset directory | Per-subject inference results (accuracy, precision, recall, F1, specificity) |

## Dependencies

```bash
# Core
python=3.10
tensorflow>=2.10
numpy>=1.21
scipy>=1.7
pandas>=1.3

# Visualization & metrics
matplotlib>=3.4
seaborn>=0.11
scikit-learn>=1.0
pingouin>=0.5

# Utilities
tqdm
statsmodels
```

## Citation

If you use this framework in your research, please cite:

```
@article{mohamed2026wearable,
    title={Wearable Interface for Real-time Gait Phase Recognition using Sensor Networks},
    author={Mohamed, Samer A. and Martinez-Hernandez, Uriel},
    journal={Applied Soft Computing},
    year={2026}
}
```

Data description available at: https://doi.org/10.15125/BATH-01425

## Author

**Samer A. Mohamed**  
University of Bath, 2025  
Email: sa2930@bath.ac.uk

## License

MIT License 
```
