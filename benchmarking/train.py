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
from util.benchmarkTrain import BMtrainer

def main():

    # create command-line argument parser for benchmark training
    parser = argparse.ArgumentParser(description="input training dataset and desired training parameters")

    # root path: subjects' directory
    _default_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "BLISS_inference"))
    parser.add_argument('--path', type=str, help="path to root folder containing training set", default=_default_dir)
    
    # model type
    parser.add_argument('--model', type=str, help="type of training model: lstm, cnn-lstm, convGRU, gnn, transformer", default="lstm")

    # frequency ratio
    parser.add_argument('--ratio', type=str, help="ratio: training freq. / test freq.", default="1")
    
    args, _ = parser.parse_known_args()

    # create an instance of the training class
    trainer = BMtrainer(path=args.path, method=args.model, fs_ratio=float(args.ratio))
    
    # train the model
    trainer.train_model_kfolds()

if __name__ == "__main__":
    main()