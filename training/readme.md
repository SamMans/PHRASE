```markdown
# PHRASE Training Framework (MATLAB)

This directory contains the MATLAB implementation of the **PHRASE (Probabilistic Heuristic-based Recognition with Adaptive Sequential Estimation)** system for gait phase recognition. PHRASE combines heuristic rules with Bayesian inference and an artificial neural network (ANN) for robust, real-time gait phase classification.

## Overview

PHRASE is a hybrid approach that integrates:

- **Heuristic detection**: Rule-based identification of gait events (peaks, zero-crossings, etc.)
- **Prior knowledge extraction**: Gaussian distribution models of biomechanically-anchored features
- **Bayesian inference**: Probabilistic fusion of heuristic detections with prior knowledge
- **Post-heuristics**: Refinement rules for precise transition timing
- **ANN classifier**: Time-domain feature-based neural network for phase classification

## Repository Structure

```
```
phrase_train/
├── trainAll.m                 # Main training script (runs all subjects)
├── PHRASE_Train.m             # Core training function for PHRASE
└── README.md                  # This file
```

## File Descriptions

### `trainAll.m`
Master training script that iterates through all subjects in the dataset and trains PHRASE models for each unseen subject (k-fold cross-validation).

**Usage:**
```matlab
% Edit the dataset name in the script if needed
% Then run:
trainAll
```

**What it does:**
- Scans the `BLISS_training` directory for CSV files
- Extracts unique subject codes
- Loops through each subject, treating them as the unseen test subject
- Calls `PHRASE_Train` for each fold

### `PHRASE_Train.m`
Main training function that:
1. Segments the dataset into training, validation, and test sets
2. Extracts raw prior features (biomechanically-anchored)
3. Trains Gaussian distribution models for each transition type
4. Extracts time-domain features for ANN training
5. Trains the phase classification neural network
6. Saves model parameters to a JSON file

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `ds` | Dataset folder name |
| `unseen_code` | Subject code to treat as unseen test subject |
| `activity` | Activity type ('W', 'RA', 'RD', 'SA', 'SD') |
| `seq_sz` | Sequence size (redundant, for benchmark comparison) |

**Note:** Currently the project supports only the walking activity 'W'

## Data Format Requirements

### Input CSV Files
Each CSV file must contain the following columns:

| Column | Description |
|--------|-------------|
| `Right_Shank_Ax` | Right shank accelerometer (x-axis) |
| `Right_Shank_Az` | Right shank accelerometer (z-axis) |
| `Right_Shank_Gy` | Right shank gyroscope (y-axis) |
| `Left_Shank_Ax` | Left shank accelerometer (x-axis) |
| `Left_Shank_Az` | Left shank accelerometer (z-axis) |
| `Left_Shank_Gy` | Left shank gyroscope (y-axis) |
| `Mode` | Activity mode (1 = walking) |
| `phase` | Gait phase label (0-6) |

### File Naming Convention
- Format: `ABXXX_circuitYY.csv` (e.g., `AB001_circuit01.csv`)
- Subject code: characters 3-5 (e.g., `001`)

### Metadata File (`metadata.json`)
Located in the dataset directory:
```json
{
    "signal_parameters": {
        "sampling_frequency": 250
    },
    "gait_phases": {
        "phase_names": ["LR", "MST", "TS", "PSW", "ISW", "MSW", "TSW"]
    }
}
```

## Gait Phase Codes

| Phase | Walking | Ramp/Stair Ascent | Ramp/Stair Descent |
|-------|---------|-------------------|---------------------|
| 0 | LR (Loading Response) | WA (Weight Acceptance) | WA (Weight Acceptance) |
| 1 | MST (Mid-Swing) | PU (Pull-Up) | FCO (Foot Clearance) |
| 2 | TS (Terminal Swing) | FCO (Foot Clearance) | CL (Curb Clearance) |
| 3 | PSW (Pre-Swing) | FCL (Foot Contact) | LP (Limb Placement) |
| 4 | ISW (Initial Swing) | FP (Forward Progression) | FP (Forward Progression) |
| 5 | MSW (Mid-Swing) | - | - |
| 6 | TSW (Terminal Swing) | - | - |

## Training Pipeline

### Step 1: Dataset Segmentation (`DS_Segment`)
- Splits data into training (60% of seen subjects), validation (40%), and test (unseen subject)
- Extracts continuous gait bouts for the specified activity

### Step 2: Prior Feature Extraction (`Prior_Raw_Extract`)
- Filters gyroscope signals (10 Hz low-pass Butterworth filter)
- Detects heuristic events (peaks, zero-crossings)
- Extracts time-normalized raw features between anchor events
- Labels features as true/false based on ground truth phase transitions

### Step 3: Prior Distribution Training (`Prior_Train`)
- Standardizes raw features (z-score)
- Applies PCA to reduce dimensionality
- Fits diagonal Gaussian distributions for true and false transitions
- Saves distribution parameters (mean, variance, projection matrix)

### Step 4: ANN Feature Extraction (`NN_Pprocess`)
- Extracts 9 time-domain features per window:
  - Mean, median, standard deviation
  - Minimum, maximum
  - Initial and final values
  - Mean Absolute Value (MAV)
  - Waveform Length (WL)

### Step 5: ANN Training (`PNN_train`)
- Single hidden layer neural network (100 units)
- Tanh activation for hidden layer, sigmoid for output
- Regularized cross-entropy loss
- Line search optimization with Wolfe-Powell conditions
- Early stopping based on validation performance

### Step 6: Model Saving
- Saves all trained parameters to a JSON file
- Output location: `../checkpoints/phrase_model_<activity>_<subject>_<fs>Hz.json`

## Output Model Format

The trained model JSON file contains:

| Field | Description |
|-------|-------------|
| `W` | ANN weights and biases |
| `D` | Prior distribution parameters per phase |
| `fsample` | Sampling frequency |
| `win_size` | Window size (samples) |
| `multiplier` | Peak detection prominence multiplier |
| `seq_sz` | Sequence size (for benchmark compatibility) |
| `phases` | Gait phase labels |
| `modalities` | Sensor modalities used |

## Helper Functions

| Function | Purpose |
|----------|---------|
| `Prior_Preprocess` | Apply standardization and PCA to raw features |
| `Temp_Normalize` | Temporal normalization of signals to 100 points |
| `diagonal_mvnpdf_vectorized` | Fast multivariate Gaussian PDF with diagonal covariance |
| `NumPCA` | Determine optimal number of PCA components |
| `augment_features` | Generate synthetic features for undersampled classes |

## Usage Example

### Basic Training
```matlab
% Train PHRASE for walking activity on a specific subject
PHRASE_Train("BLISS_training", "001", "W", 10);
```

### Full Cross-Validation
```matlab
% Run training for all subjects in the dataset
trainAll
```

### Monitor Training Progress
The training script displays:
- Prior accuracy per phase
- Overall prior accuracy, recall, precision, specificity
- ANN validation and testing metrics (F1, precision, recall, specificity)
- Training/validation cost and accuracy plots

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `*.mat` | Dataset directory | Cached segmentation and features (per subject) |
| `phrase_model_*.json` | `../checkpoints/` | Trained model parameters |

## Dependencies

- MATLAB R2020b or later
- Signal Processing Toolbox (for `findpeaks`, `filtfilt`, `butter`)
- Statistics and Machine Learning Toolbox (for `pca`, `zscore`)
- Curve Fitting Toolbox (for `interp1`)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mohamed2026wearable,
    title={Wearable Interface for Real-time Gait Phase Recognition using Sensor Networks},
    author={Mohamed, Samer A. and Martinez-Hernandez, Uriel},
    journal={Applied Soft Computing},
    year={2026}
}
```

Data available at: https://doi.org/10.15125/BATH-01425

**Note**: The source data is only shown for recognition. The source data was processed and included already in the repo, so don't attempt to download the source data from the link above.

## Author

**Samer A. Mohamed**  
University of Bath, 2025  
Email: sa2930@bath.ac.uk

Personal Email: samermansour1994@gmail.com

## Notes

- The training process can be time-consuming depending on dataset size
- The `augment_features` function synthesizes data for undersampled transitions
- Prior knowledge is only applied in 'full' mode (other modes: 'ANN', 'HANN')
- The `seq_sz` parameter is primarily for compatibility with benchmark models

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot find metadata.json" | Ensure dataset directory contains the metadata file |
| "File not found" | Check that CSV files follow the naming convention |
| Empty bins warning | This is normal for signals with no peaks detected |
| PCA numerical issues | Large feature vectors may cause instability; reduce `win_size` |

## License

MIT License 
