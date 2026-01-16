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

def main():

    # create command-line argument parser for benchmark inference
    parser = argparse.ArgumentParser(description="input inference dataset, desired inference model and other inference parameters.")

    # root path: subjects' directory
    _default_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "BATH_inference"))
    parser.add_argument('--path', type=str, help="path to root folder containing inference set", default=_default_dir)

    # root path: models' directory
    _default_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    parser.add_argument('--pretrained', type=str, help="path to root folder containing pretrained models", default=_default_dir)
    
    # model type
    parser.add_argument('--model', type=str, help="type of training model: lstm, cnn-lstm, convGRU, gnn, transformer", default="lstm")

    # evaluation type
    parser.add_argument('--eval', type=str, help="type of evaluation: training, validation or testing", default="test")
    
    args, _ = parser.parse_known_args()

    # create an instance of the training class
    inferencer = BMinf(Mparam=args.pretrained, infMode=args.eval, PHRASE_mode='HANN only')
    inferencer.set_path(Dset=args.path)
    inferencer.set_test_sub(u="112")
    inferencer.get_inf_metrics(args.model)

if __name__ == "__main__":
    main()