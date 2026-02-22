#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Created on Sat Dec 23 02:28:05 2023

@author: Samer A. Mohamed
@email: sa2930@bath.ac.uk
@affiliation: University of Bath, 2025

"""

import argparse
import os
from util.benchmarkInf import BMinf
from pathlib import Path
import json

def main():

    # create command-line argument parser for benchmark inference
    parser = argparse.ArgumentParser(description="input inference dataset, desired inference model and other inference parameters.")

    # root path: subjects' directory
    parser.add_argument('--dataset', type=str, help="inference set name", default="BLISS_inference")

    # root path: models' directory
    _default_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    parser.add_argument('--pretrained', type=str, help="path to root folder containing pretrained models", default=_default_dir)
    
    # model type
    parser.add_argument('--model', type=str, help="type of training model: lstm, cnn-lstm, convGRU, gnn, transformer", default="lstm")

    # ablation type (for phrase only)
    parser.add_argument('--abl', type=str, help="type of ablation: ANN, HANN or full", default="full")

    # evaluation type
    parser.add_argument('--eval', type=str, help="type of evaluation: training, validation or testing", default="test")
    
    args, _ = parser.parse_known_args()

    # Contruct the full path to the dataset
    dataset_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", args.dataset))

    # Adjust the severity indicator
    if 'severe' in args.dataset:
        severity_indicator = True
    else:
        severity_indicator = False

    # create an instance of the training class
    inferencer = BMinf(Mparam=args.pretrained, infMode=args.eval, PHRASE_mode=args.abl, severe = severity_indicator)
    inferencer.set_path(Dset=dataset_dir)

    # get the list of all unique subjects within target path
    csv_files = Path(dataset_dir).glob("*.csv")
    subject_ids = {f.stem[2:5] for f in csv_files}

    # Calculate metrics for all subjects, and save in dataset directory
    results = {
        'accuracy': [],
        'precision': {'LR': [], 'MST': [], 'TS': [], 'PSW': [], 'SW': []},
        'recall': {'LR': [], 'MST': [], 'TS': [], 'PSW': [], 'SW': []},
        'f1': {'LR': [], 'MST': [], 'TS': [], 'PSW': [], 'SW': []},
        'specificity': {'LR': [], 'MST': [], 'TS': [], 'PSW': [], 'SW': []}
    }

    # Class labels in order
    class_labels = ['LR', 'MST', 'TS', 'PSW', 'SW']

    for id in subject_ids:
        inferencer.set_test_sub(u=id)
        overall_accuracy, precision, recall, f1, specificity = inferencer.get_inf_metrics(args.model)
        
        # Append overall accuracy
        results['accuracy'].append(overall_accuracy)
        
        # Append per-class metrics using loop
        for i, label in enumerate(class_labels):
            results['precision'][label].append(precision[i])
            results['recall'][label].append(recall[i])
            results['f1'][label].append(f1[i])
            results['specificity'][label].append(specificity[i])

    # Save results to JSON file
    if args.model != 'phrase':
        result_path = os.path.join(dataset_dir, args.model+'_results.json')
    else:
        result_path = os.path.join(dataset_dir, args.model+'_'+args.abl+'_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()