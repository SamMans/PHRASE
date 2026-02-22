"""
SIMPLE PAIRED WILCOXON TEST WITH EFFECT SIZES
Using pingouin library
"""

import numpy as np
import pandas as pd
import pingouin as pg
import argparse
from pathlib import Path
import json
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ds1', '--ds2', '--ds3', '--ds4', '--ds5', '--ds6', 
                    dest='datasets', action='append', help="dataset names")
parser.add_argument('--metric', default='accuracy')
parser.add_argument('--phase', default='LR')
args = parser.parse_args()

# Resources path
resources_dir = Path(__file__).parent.parent / 'resources'

# Initialize benchmarks
benchmarks = {method: [] for method in 
              ['phrase', 'lstm', 'cnn-lstm', 'convgru', 'gnn', 'transformer']}

# Load data
for dataset in filter(None, args.datasets or []):
    for method in benchmarks:
        file_path = resources_dir / dataset / f"{method}_results.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            values = data[args.metric] if args.metric == 'accuracy' else data[args.metric][args.phase]
            benchmarks[method].extend(values)

# Convert to numpy arrays
benchmarks = {k: np.array(v) for k, v in benchmarks.items()}

# ==================== RUN WILCOXON TESTS ====================
print("=" * 60)
print("PAIRED WILCOXON TEST: PHRASE vs. BENCHMARKS")
print("=" * 60)

benchmarks = {
    'LSTM': LSTM,
    'CNN-LSTM': CNN_LSTM,
    'Transformer': Transformer
}

results = {}

for name, benchmark_data in benchmarks.items():
    print(f"\n--- PHRASE vs. {name} ---")
    
    # Run Wilcoxon test with pingouin
    result = pg.wilcoxon(PHRASE, benchmark_data, alternative='two-sided')
    
    # Extract key statistics
    w_stat = result['W-val'].values[0]
    p_val = result['p-val'].values[0]
    effect_size = result['RBC'].values[0]  # Rank-biserial correlation
    cles = result['CLES'].values[0]        # Common Language Effect Size
    
    # Store results
    results[name] = {
        'W_statistic': w_stat,
        'p_value': p_val,
        'effect_size_r': effect_size,
        'CLES': cles
    }
    
    # Print results
    print(f"W statistic: {w_stat:.1f}")
    print(f"p-value: {p_val:.6f}")
    print(f"Effect size (RBC): {effect_size:.3f}")
    print(f"CLES: {cles:.3f}")
    
    # Significance interpretation
    if p_val < 0.001:
        print("→ p < 0.001 (HIGHLY SIGNIFICANT)")
    elif p_val < 0.01:
        print("→ p < 0.01 (VERY SIGNIFICANT)")
    elif p_val < 0.05:
        print("→ p < 0.05 (SIGNIFICANT)")
    else:
        print("→ NOT SIGNIFICANT")
    
    # Effect size interpretation
    if abs(effect_size) >= 0.5:
        print("→ Large effect size")
    elif abs(effect_size) >= 0.3:
        print("→ Medium effect size")
    elif abs(effect_size) >= 0.1:
        print("→ Small effect size")
    else:
        print("→ Negligible effect size")

# ==================== SIMPLE SUMMARY TABLE ====================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary_df = pd.DataFrame(results).T
print(summary_df.round(4))

# ==================== EXPORT TO CSV ====================
summary_df.to_csv('wilcoxon_results.csv')
print("\nResults saved to 'wilcoxon_results.csv'")