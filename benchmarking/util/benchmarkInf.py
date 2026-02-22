#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 19 20:51:04 2024

@author: Samer A. Mohamed
@email: sa2930@bath.ac.uk
@affiliation: University of Bath, 2025

"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, Conv2D, MaxPooling2D, Flatten, Dropout, Conv1D, Lambda, Concatenate, Multiply, Add, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import numpy as np
import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.signal import find_peaks, butter, filtfilt
import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import warnings

class phrase_inf:
    """

    This class classifies gait phases using the Heuristic-Bayesian Inference System.
    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        infinity (int): system's maximum integer number.
        D (dictionary): prior probability distribution parameters.
        win_sz (int): signal window size (samples per window).
        seq_sz (int): redundant parameter representing the number of windows to 
            ignore at the beginning of a bout for external evaluation scripts.
        mul (int): window multiplier for peak finding.
        W (nd.array): ANN weights.
        b, a (float): butterworth filter parameters.
        phases (list of strings): gait phases for walking.
        candidacy (nd.array of bool): heuristic candidacy status for all windows.
        likelihood (list): probability score for each phase-window pair.
        anchor_status (list): boolean anchor status for each window (anchor if true).
        modalities (str): Sensory modalities.
        severe (boolean): Severity indicator

    Developer/s:
        Samer A. Mohamed.
        
    """    
    def __init__(self, fs, PrModel, mode='full', severe=False):
        """

        Class constructor: initializes class parameters.
    
        Args:
            fs (float): Sampling frequency.
            PrModel (str): PHRASE model file path.
            mode (str): The mode of operation of PHRASE ('ANN', 'HANN', and 'full')
            severe (boolean): Severity indicator
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """
        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            else:
                # Define class attributes
                with open(PrModel, 'r') as file:
                    json_text = file.read()

                # Decode the JSON file
                hParameters = json.loads(json_text)                                                                                                             # Hyperparameters
                
                # Assign class parameters
                self._operation_mode = mode                                                                                                                     # PHRASE's operation mode
                self.infinity = sys.maxsize
                self.D = {'MST' : {'true':[], 'false':[]}, 'TS' : {'true':[], 'false':[]}, \
                          'ISW' : {'true':[], 'false':[]}, 'MSW' : {'true':[], 'false':[]}, \
                            'TSW' : {'true':[], 'false':[]}}                                                                                                    # Prior distribution parameters
                for key in hParameters['D'].keys():
                    for subkey, value in hParameters['D'][key].items():
                        if subkey not in ['true', 'false']:
                            self.D[key][subkey] = np.array(value) # Convert parameters to numpy arrays
                        else:
                            self.D[key][subkey] = {k: np.array(v) for k, v in hParameters['D'][key][subkey].items()}                                            # Convert parameters to numpy arrays
                self.win_sz = round(fs*0.03)                                                                                                                    # Window size
                self.seq_sz = hParameters['seq_sz']                                                                                                             # Sequnce size
                self.mul = hParameters['multiplier']                                                                                                            # Peak range multiplier
                self.W = []                                                                                                                                     # ANN weights
                for elem in range(len(hParameters['W'])):
                    self.W.append({k: np.array(v) if k != 'fn' else v for k, v in hParameters['W'][elem].items()})                                              # Convert parameters to numpy arrays 
                self.modalities = hParameters['modalities']                                                                                                     # Sensor modalities   
                if severe:
                    # Lower angular velocity thresholds for slow impaired patients
                    self.threshold_negative = 0.5
                    self.threshold_positive = 0.2
                else:
                    # Higher thresholds for healthy or mildly impaired patients
                    self.threshold_negative = 1.5
                    self.threshold_positive = 0.5
                    
                # Develop heuristics filter
                nyquist_freq = 0.5 * fs                                                                                                                         # Sampling rate: fs Hz
                normalized_cutoff = 10 / nyquist_freq                                                                                                           # Cutoff frequency: 10 Hz
                self.b, self.a = butter(4, normalized_cutoff, btype='low', analog=False)                                                                        # 4th order butterworth low pass
        
                # Phase labels
                self.phases = ['LR', 'MST', 'TS', 'PSW', 'ISW', 'MSW', 'TSW']
        
                # Initialize PHRASE dynamic parameters
                self.feedback = [] # Cumulative feedback array
                self.candidacy = np.array([],dtype=bool).reshape((7,0)) # Transition candidacy array
                self.log_likelihood = []; self.anchor_status = [] # ANN likelihood and anchor status arrays
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.win_sz (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """
        return self.win_sz

    def get_seq_sz(self):
        """

        Get sequence size: returns the model's sequence size.

        Args: 
            N/A.
            
        Returns:
            self.seq_sz (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """
        return self.seq_sz
    
    def shut_down(self):
        """

        Bayesian prediction reset function: resets the bayesian memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """
        # Reset belief arrays
        self.feedback = []; self.candidacy = np.array([],dtype=bool).reshape((7,0))
        self.log_likelihood = []; self.anchor_status = []
    
    def findpeaksamer(self, x, MinPeakHeight, MinPeakDistance):
        """

        Finds positive peaks in the signal.

        Args:
            x : numpy array
                Signal vector.
            MinPeakHeight : float
                Minimum peak height.
            MinPeakDistance : float
                Minimum allowable distance between two peaks.

        Returns:
            peak_locs : numpy array
                Peak indices in the given signal array.
                
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.
            
        """
        # Find indices of unique samples
        x = x + np.random.uniform(-1e-10, 1e-10, size=x.shape) # Add ultra-small white noise
        _, uniqueIdcs = np.unique(x, return_index=True)
        uniqueIdcs = np.sort(uniqueIdcs)  # Ensure indices are sorted

        # Get points that are greater than immediate neighbors & greater than MinPeakHeight
        diff_left = np.concatenate(([False], np.diff(x[uniqueIdcs]) > 0))  # Left difference
        diff_right = np.diff(x[uniqueIdcs][::-1]) > 0  # Right difference (flipped)
        diff_right = np.concatenate((diff_right[::-1], [False]))  # Flip back and pad
        peak_locs = uniqueIdcs[diff_left & diff_right & (x[uniqueIdcs] >= MinPeakHeight)]

        # Further filter to get points that are the highest peaks within MinPeakDistance
        if len(peak_locs) > 0:
            # Create a distance matrix
            distance_matrix = np.abs(peak_locs[:, None] - peak_locs)
            height_matrix = x[peak_locs][:, None] - x[peak_locs]

            # Find peaks that are too close and not the highest
            to_remove = np.any((distance_matrix < MinPeakDistance) & (height_matrix < 0), axis=1)
            peak_locs = peak_locs[~to_remove]

        return peak_locs
    
    def fast_linear_interp(self, signal, new_length=100):
        """

        Fast linear interpolation of a signal to a new length.

        Args:
            signal (np.array): Input signal to interpolate.
            new_length (int): Desired length of the interpolated signal.

        Returns:
            np.array: Interpolated signal of length `new_length`.

        Developer/s:
            DeepSeek AI.

        """
        original_length = len(signal)
        if original_length == 0:
            return np.zeros(new_length)
        
        # Generate the original and new x-axis
        x_original = np.linspace(0, 100, original_length)
        x_new = np.linspace(0, 100, new_length)
        
        # Perform linear interpolation
        return np.interp(x_new, x_original, signal)

    def Prior_fn(self, feat, phase):
        """

        Prior likelihood calculation function.

        Args:
            feat : numpy array
                Raw features.
            phase : str
                The phase for which to compute the prior likelihood.

        Returns:
            prior_est : numpy array
                Prior likelihood estimate.
                
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.

        """
        # Process features using PCA parameters
        feat = ((feat - self.D[phase]['mu_raw']) / self.D[phase]['sigma_raw']) @ self.D[phase]['proj']
        feat = feat[:, :self.D[phase]['num_components']]  # Processed features

        # Conditional probability estimate given false/true transition
        prior_est = np.vstack((self.diagonal_mvnpdf_vectorized(feat, self.D[phase]['true']['mu'],
            self.D[phase]['true']['cov']), self.diagonal_mvnpdf_vectorized(feat, self.D[phase]['false']['mu'],
            self.D[phase]['false']['cov'])))

        return prior_est

    def diagonal_mvnpdf_vectorized(self, X, mu, sigma_sq):
        """

        Compute multivariate Gaussian PDF with diagonal covariance matrix.
        
        Parameters
        ----------
        X : numpy.ndarray
            An array containing input features [n_samples, n_features]
        mu : numpy.ndarray
            Feature means [n_features,]
        sigma_sq : numpy.ndarray
            Feature variances [n_features,] (diagonal of covariance matrix)
        
        Returns
        -------
        pdf : numpy.ndarray
            Probability density for each sample [n_samples,]

        Developer/s:
            Samer A. Mohamed.

        """
        
        # Get a vector of variances
        sigma_sq = np.diag(sigma_sq).reshape(1, -1)
        
        d = X.shape[1]  # Number of features
        sigma = np.sqrt(sigma_sq)  # Standard deviations
        
        # Normalization constant
        norm_const = 1.0 / ((2 * np.pi) ** (d / 2.0) * np.prod(sigma))
        
        # Centered data
        # Reshape mu to (1, n_features) for broadcasting
        X_centered = X - mu.reshape(1, -1)
        
        # Quadratic form for ALL points at once
        # (X - μ)^T Σ^{-1} (X - μ) = sum((x_i - μ_i)^2 / σ_i^2)
        quad_form = np.sum((X_centered ** 2) / sigma_sq, axis=1)
        
        # PDF for all points
        pdf = norm_const * np.exp(-0.5 * quad_form)
        
        return pdf

    def activ(self, z, fn):
        """

        Compute activation function.

        Args:
            z : numpy array
                An array containing weighted sums of inputs.
            fn : str
                Activation function name ("sigmoid", "tanh", "relu", or "leaky relu").

        Returns:
            a : numpy array
                An array containing activations.
                
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.

        """
        if fn == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-z))
        elif fn == "tanh":
            a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif fn == "relu":
            a = np.maximum(0, z)
        elif fn == "leaky relu":
            a = np.maximum(0.01 * z, z)
        else:
            raise ValueError(f"Unsupported activation function: {fn}")
        
        return a
    
    def ANN(self, X):
        """

        Compute prediction.

        Args:
            X : numpy array
                An array containing input features.

        Returns:
            p : numpy array
                An array containing output predictions.
                
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.

        """
        # Time-domain features
        X_features = np.hstack([
            np.mean(X, axis=0),
            np.median(X, axis=0),
            np.std(X, axis=0),
            np.min(X, axis=0),
            np.max(X, axis=0),
            X[0, :],
            X[-1, :],
            np.mean(np.abs(X), axis=0),
            np.sum(np.abs(np.diff(X, axis=0)), axis=0)
        ])

        # Compute input activations
        num_hL = len(self.W) - 1  # Number of hidden layers
        A = [{'a': None, 'da': None} for _ in range(num_hL + 2)]  # Activation storage

        A[0]['a'] = X_features # Input layer activations (column vector)
        A[0]['da'] = np.zeros_like(A[0]['a'])  # Derivative of input layer units (initially zero)

        # Forward propagation
        for k in range(1, len(A)):
            z = np.dot(self.W[k - 1]['w'], A[k - 1]['a']) + self.W[k - 1]['b']
            A[k]['a'] = self.activ(z, self.W[k - 1]['fn'])

        # Softmax to normalize
        p = A[-1]['a'] / np.sum(A[-1]['a'])

        return p
    
    def generate_trajs(self, hit_subset):
        """

        Generates potential phase trajectories in the presence of initial parent phase(s).

        Args:
            hit_subset : list or numpy array
                An array containing phases of heuristic hits.

        Returns:
            sequences : list of lists
                A list containing potential trajectories starting with the independent
                parent index in `hit_subset`.
                
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.

        """
        # Preallocate empty sequence list
        sequences = []

        # Check for immediate children
        parent_hit = hit_subset[0]
        if parent_hit != 6:
            children_idx = [i for i, x in enumerate(hit_subset) if x == parent_hit + 1]
        else:
            children_idx = [i for i, x in enumerate(hit_subset) if x == 0]  # Wrap-around case

        # Loop over all immediate children
        if children_idx:
            # If children exist, loop over them
            for child_count in range(len(children_idx)):
                # Recursion
                child_sequences = self.generate_trajs(hit_subset[children_idx[child_count]:])
                
                # Append children sequences one by one after recursion
                offset = children_idx[child_count]  # Precompute the offset for alignment
                for c_seq in child_sequences:
                    # Re-align children indices with current parent
                    aligned_seq = [x + offset for x in c_seq]
                    sequences.append([0] + aligned_seq)  # Append child sequence

        # If no children exist, then current hit finished the trajectory
        # If children do exist, then finalize by adding trivial trajectory
        sequences.append([0])

        return sequences
    
    def predict(self, signal):
        """

        Hybrid bayesian prediction function: predicts the gait phase
            based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            prediction (int): Phase prediction.
            confidence (float): Path probability.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        try:
            # Check if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.win_sz:
                raise ValueError(f"Signal length is not equal to the window size which is {self.win_sz}.")
            
            """ Processing Start : Appending New Windows """
            # Appending new observations
            self.feedback.extend(signal[self.modalities].values.tolist())
            windows_num = int(np.floor(len(self.feedback)/self.win_sz)) # Number of windows under analysis
            signal_length = int(windows_num*self.win_sz) # Length of the signal under analysis
            past_windows_num  = len(self.log_likelihood) # Past number of windows from old analyses
            
            # Extract gyro signals
            signal = {} # Dictionary of individual signals
            signal['rGyro'] = np.array(self.feedback)[:signal_length, 2] # Right gyro signal
            signal['lGyro'] = np.array(self.feedback)[:signal_length, 5] # Left gyro signal

            # Initialize anchor status for new windows, then append
            for _ in range(past_windows_num, windows_num):
                self.anchor_status.append([False, False]) # Initially false, till PHRASE decides otherwise

            # Initialize candidacy status for new windows
            if self.candidacy.size == 0:
                new_column = np.ones(7, dtype=bool) # First window status is unknown, so assume all hits occur
            else:
                new_column = np.zeros(7, dtype=bool) # New window status is False, until proved otherwise by heuristics
            self.candidacy = np.hstack((self.candidacy, new_column[:, np.newaxis], \
                                        np.zeros((7,windows_num-past_windows_num-1), dtype=bool)))  # Append the column

            """ Heuristics : Selecting Transition Candidates """
            # Filter the gyro signals
            if signal['rGyro'].shape[0] > 15:  # Check if the number of rows (samples) is greater than 12
                fRgyro = filtfilt(self.b, self.a, signal['rGyro'])
                fLgyro = filtfilt(self.b, self.a, signal['lGyro'])
            else:
                # No filtration if sample size is very small
                fRgyro = signal['rGyro']  
                fLgyro = signal['lGyro']

            # Current scan heuristic hits
            Hits = np.zeros((7, windows_num), dtype=bool) # Binary hits array
            prom_win_sz = self.mul * self.win_sz # Multiplier
            Rneg_locs, _ = find_peaks(-fRgyro, height=self.threshold_negative, 
                        distance=min(len(fRgyro) - 2, prom_win_sz)) # Right negative peaks
            Rpos_locs, _ = find_peaks(fRgyro, height=self.threshold_positive, 
                        distance=min(len(fRgyro) - 2, prom_win_sz)) # Right positive peaks
            if len(Rneg_locs) > 0:
                # Right negative to positive zero crossings (ZC) with conditions
                # 1- Must be preceded by at least one negative peak
                Rnp_zc_locs = np.where(np.diff(np.sign(signal['rGyro'])) > 0)[0]
                A = Rnp_zc_locs[:, None] - Rneg_locs; A[A <= 0] = self.infinity  # Progressive distances between Rnp_zc_locs and Rneg_locs
                Rnp_zc_locs = Rnp_zc_locs[np.any(A != self.infinity, axis=1)] # Keep zero crossings with finite distance from a peak (1st condition)
                if len(Rnp_zc_locs) > 0:
                    A_min = np.min(Rnp_zc_locs[:, None] - Rneg_locs, axis=1)  # Find the smallest distance from a peak for each ZC
                    B = Rnp_zc_locs[:, None] - Rnp_zc_locs  # Pairwise distances between zero-crossings (ZC)
                    Rnp_zc_locs = Rnp_zc_locs[np.all((B <= 0) | ((B > 0) & (B > A_min[:, None])), axis=1)]  # Filter ZC's that occur after other ZC's (2nd condition)
            else:
                Rnp_zc_locs = np.array([])  # No valid zero crossings
            Rpn_zc_locs = np.where(np.diff(np.sign(signal['rGyro'])) < 0)[0] # Right positive to negative ZC
            Lneg_locs, _ = find_peaks(-fLgyro, height=self.threshold_negative, 
                        distance=min(len(fLgyro) - 2, prom_win_sz)) # Left negative peaks
            Lpos_locs, _ = find_peaks(fLgyro, height=self.threshold_positive, 
                        distance=min(len(fLgyro) - 2, prom_win_sz)) # Left positive peaks
            if len(Lneg_locs) > 0:
                # Left negative to positive zero crossings (ZC) with conditions
                # 1- Must be preceded by at least one negative peak
                Lnp_zc_locs = np.where(np.diff(np.sign(signal['lGyro'])) > 0)[0]
                A = Lnp_zc_locs[:, None] - Lneg_locs; A[A <= 0] = self.infinity  # Progressive distances between Lnp_zc_locs and Lneg_locs
                Lnp_zc_locs = Lnp_zc_locs[np.any(A != self.infinity, axis=1)] # Keep zero crossings with finite distance from a peak (1st condition)
                if len(Lnp_zc_locs) > 0:
                    A_min = np.min(Lnp_zc_locs[:, None] - Lneg_locs, axis=1)  # Find the smallest distance from a peak for each ZC
                    B = Lnp_zc_locs[:, None] - Lnp_zc_locs  # Pairwise distances between zero-crossings (ZC)
                    Lnp_zc_locs = Lnp_zc_locs[np.all((B <= 0) | ((B > 0) & (B > A_min[:, None])), axis=1)]  # Filter ZC's that occur after other ZC's (2nd condition)
            else:
                Lnp_zc_locs = np.array([])  # No valid zero crossings
            Hits[6,np.floor(Rneg_locs/self.win_sz).astype(int)] = True; Hits[4,np.floor(Rpos_locs/self.win_sz).astype(int)] = True
            Hits[0,np.floor(Rnp_zc_locs/self.win_sz).astype(int)] = True; Hits[5,np.floor(Rpn_zc_locs/self.win_sz).astype(int)] = True 
            Hits[2,np.floor(Lneg_locs/self.win_sz).astype(int)] = True; Hits[1,np.floor(Lpos_locs/self.win_sz).astype(int)] = True 
            Hits[3,np.floor(Lnp_zc_locs/self.win_sz).astype(int)] = True
            self.candidacy[[0,3,5],1:] = self.candidacy[[0,3,5],1:] | Hits[[0,3,5],1:] # Zero crossings are OR'ed with their past status
            self.candidacy[[1,2,4,6],1:] = Hits[[1,2,4,6],1:] # Peaks are renewed regardless of past status
            l_anchor_phase = 3 # Left anchor index

            """Prior Knowledge : Compute Prior Probabilities"""
            # Initialize prior belief arrays
            prior_prob_given_true = np.ones_like(self.candidacy, dtype=float)
            prior_prob_given_false = np.ones_like(self.candidacy, dtype=float)
            if self._operation_mode != 'HANN':
                prior_prob_given_true[[0,2,3,6], :] = 5
                r_anchor_locs = np.where(np.array(self.anchor_status)[:, 0])[0].astype(int) # anchor window location
                l_anchor_locs = np.where(np.array(self.anchor_status)[:, 1])[0].astype(int)
                r_anchor_idx = np.round((r_anchor_locs + 0.5) * self.win_sz).astype(int) - 1 # anchor sample index
                l_anchor_idx = np.round((l_anchor_locs + 0.5) * self.win_sz).astype(int) - 1
                
                # Estimate maximum number of features and phase-window pairs
                max_feats = windows_num  # Maximum possible number of features
                max_pairs =  windows_num # Maximum possible number of phase-window pairs
                
                # Preallocate arrays
                Feats = np.zeros((max_feats, 100))  # Each feature is 100 elements
                phase_win_pair = np.zeros((max_pairs, 2), dtype=int)  # Each pair is [phase_num, cand_win]
                feat_counter = 0
                pair_counter = 0
                
                # Process right leg anchors
                for idx in range(len(r_anchor_locs)):
                    anchored_phases_num = np.array([4]) # Type 5 points (ISW preheuristics) associated with r. anchor
                    start_idx = r_anchor_locs[idx] + 1 # Start of the right anchor territory in the array
                    next_anchor = np.argmax(self.candidacy[0, start_idx:]) + start_idx if np.any(self.candidacy[0, start_idx:]) else self.candidacy.shape[1]
                    anchor_range = np.arange(start_idx, next_anchor) # Range of right anchor territory in the array
                    phase_num, cand_win = np.where(self.candidacy[anchored_phases_num][:, anchor_range]) # Search for assoc. candidates
                    cand_win += start_idx # Adjust candidate sample index
                    cand_idx = np.round((cand_win + 0.5) * self.win_sz).astype(int) - 1
                    phase_num = anchored_phases_num[phase_num] # Adjust phase number to match the gait phase numerical code
                
                    for assoc_cand in range(len(phase_num)):
                        feat_range = range(r_anchor_idx[idx], cand_idx[assoc_cand] + 1) # Range of the raw feature
                        if len(feat_range) >= self.win_sz:
                            norm_left = self.fast_linear_interp(fLgyro[feat_range]) # Temporally normalized feature for associated phases
                            Feats[feat_counter] = norm_left                         
                        else:
                            Feats[feat_counter] = np.rand(100)
                        feat_counter += 1
                
                        phase_win_pair[pair_counter] = [phase_num[assoc_cand], cand_win[assoc_cand]] # Add the new pair
                        pair_counter += 1
                
                # Process left leg anchors
                for idx in range(len(l_anchor_locs)):
                    anchored_phases_num = np.array([1, 5]) # Type 2 and 6 points (MST and MSW preheuristics) associated with l. anchor
                    start_idx = l_anchor_locs[idx] + 1 # Start of the left anchor territory in the array
                    next_anchor = np.argmax(self.candidacy[l_anchor_phase, start_idx:]) + start_idx if np.any(self.candidacy[l_anchor_phase, start_idx:]) else self.candidacy.shape[1]
                    anchor_range = np.arange(start_idx, next_anchor) # Range of left anchor territory in the array
                    phase_num, cand_win = np.where(self.candidacy[anchored_phases_num][:, anchor_range]) # Search for assoc. candidates
                    cand_win += start_idx # Adjust candidate sample index
                    cand_idx = np.round((cand_win + 0.5) * self.win_sz).astype(int) - 1
                    phase_num = anchored_phases_num[phase_num] # Adjust phase number to match the gait phase numerical code
                
                    for assoc_cand in range(len(phase_num)):
                        feat_range = range(l_anchor_idx[idx], cand_idx[assoc_cand] + 1) # Range of the raw feature
                        if len(feat_range) >= self.win_sz:
                            norm_right = self.fast_linear_interp(fRgyro[feat_range]) # Temporally normalized feature for associated phases
                            Feats[feat_counter] = norm_right                         
                        else:
                            Feats[feat_counter] = np.zeros(100)
                        feat_counter += 1
                
                        phase_win_pair[pair_counter] = [phase_num[assoc_cand], cand_win[assoc_cand]] # Add the new pair
                        pair_counter += 1
                
                # Trim preallocated arrays to actual size
                Feats = Feats[:feat_counter]
                phase_win_pair = phase_win_pair[:pair_counter]
                
                # Calculate prior probability
                if len(phase_win_pair) > 0:
                    for u in [1, 4, 5]:
                        WINS = phase_win_pair[(phase_win_pair[:, 0] == u) & (phase_win_pair[:, 1]>=self.candidacy.shape[1]-2), 1]
                        if len(WINS) > 0:
                            # Compute prior probabilities for fuzzy regions
                            FEATS = Feats[(phase_win_pair[:, 0] == u) & (phase_win_pair[:, 1]>=self.candidacy.shape[1]-2), :]
                            pp = self.Prior_fn(FEATS, self.phases[u]) # Compute prior based on collected features

                            # Use priors for windows that have a high reject prior probability (likely false)
                            valid_consult_idx = (pp[1, :] / (pp[0, :] + pp[1, :])) >= 0.7
                            if np.any(valid_consult_idx):
                                prior_prob_given_true[u, WINS] = pp[0, valid_consult_idx]  
                                prior_prob_given_false[u, WINS] = pp[1, valid_consult_idx]  

            """Likelihood : Apply Local Pattern Recognition"""
            for newWin in range(past_windows_num,windows_num):
                pure_ann_output = np.log(self.ANN(np.array(self.feedback[newWin*self.win_sz:(newWin+1)*self.win_sz])))
                if len(self.log_likelihood) == 0:
                    new_likelihood = pure_ann_output  # First window's likelihood
                else:
                    new_likelihood = np.array(self.log_likelihood[-1]) + pure_ann_output
                self.log_likelihood.append(new_likelihood) # Append cumulative likelihoods

            """Bayesian Inference : Fuse Beliefs"""
            # Initialize bayesian results
            log_prob_max = -self.infinity; prob_sum = 0; path = {
            "phases": [],
            "windows": [],
            }
            hit_wins, hit_phases = np.nonzero(self.candidacy.T) # Transpose to get properly sorted windows
            ind_mask = hit_wins == 0  # Mask for independent phases
            dep_mask = hit_wins != 0  # Mask for dependent phases
            ind_phases = hit_phases[ind_mask] 
            dep_phases = hit_phases[dep_mask]
            ind_wins = hit_wins[ind_mask]
            dep_wins = hit_wins[dep_mask]

            # Extract trajectories and compute path probabilities
            for ind_phase_idx in range(len(ind_phases)):
                # Extract potential trajectories
                assoc_phases = np.append(ind_phases[ind_phase_idx], dep_phases)
                assoc_wins = np.append(ind_wins[ind_phase_idx], dep_wins)
                trajectories = self.generate_trajs(assoc_phases)
                
                # Loop over trajectories
                for traj_idx in range(len(trajectories)):
                    # Extract trajectory locations
                    traj = trajectories[traj_idx]
                    traj_phases = assoc_phases[traj]
                    traj_wins = assoc_wins[traj]

                    # Compute trajectory probability
                    log_prob = np.sum(np.log(prior_prob_given_true[traj_phases, traj_wins])) + \
                        (np.sum(np.log(prior_prob_given_false[prior_prob_given_false!=1])) - \
                         np.sum(np.log(prior_prob_given_false[traj_phases, traj_wins]))) # Prior part
                    traj_wins = np.append(traj_wins, len(self.log_likelihood)) # Append last window index
                    for segment_idx in range(len(traj_phases)):
                        if traj_wins[segment_idx] != 0:
                            seg_log_prob = self.log_likelihood[traj_wins[segment_idx+1]-1][traj_phases[segment_idx]] - \
                                self.log_likelihood[traj_wins[segment_idx]-1][traj_phases[segment_idx]] # Segment probability
                        else:
                            seg_log_prob = self.log_likelihood[traj_wins[segment_idx+1]-1][traj_phases[segment_idx]]
                        if seg_log_prob.size != 0:
                            log_prob = log_prob + seg_log_prob # Posterior calculation
                    prob_sum = prob_sum + np.exp(log_prob) 
                    if log_prob_max < log_prob:
                        # Update bayesian variables (path: phases and associated windows)
                        log_prob_max = log_prob; path['phases'] = traj_phases
                        path['windows'] = traj_wins[:-1]
            confidence = np.exp(log_prob_max)/prob_sum # Probability of most likely path

            """Post-Heuristics : Specifying True Transition Locations"""
            if path['windows'][-1]==0 and not np.any(np.array(self.anchor_status[0])):
                prediction = path['phases'][-1]
            else:
                hit_range = range(max(0, path['windows'][-1]*self.win_sz - self.win_sz), \
                                  min(signal_length, path['windows'][-1] * self.win_sz + 2*self.win_sz)) # Range within hit occurs
                hit_offset = max(0, path['windows'][-1]*self.win_sz - self.win_sz) # Offset of that range from the start of the true signal
                if self.phases[path['phases'][-1]] == "LR":
                    # Post-heuristic process: is there a peak after the preheuristic hit?
                    hh = np.where(np.diff(np.sign(signal['rGyro'][hit_range])) > 0)[0] + hit_offset
                    hh = hh[0] # Index of Type 1 heuristic hit
                    prediction = 6 # Default prediction of the phase right before LR
                    uniqueIdcs = np.unique(signal['rGyro'][hh + 1 : signal_length - 1], return_index=True)[1]
                    uniqueIdcs = uniqueIdcs + hh + 1 # Indices of unique elements after preheuristic hit
                    for j in uniqueIdcs:
                        # Check which of the unique points following preheuristic hit is a peak
                        if signal['rGyro'][j] > signal['rGyro'][j - 1] and signal['rGyro'][j] > signal['rGyro'][j + 1]:
                            prediction = 0 # If a peak is found, then post-heuristic LR is detected
                            break
                elif self.phases[path['phases'][-1]] == "MST": 
                    # Post-heuristic process: find the index of the maximum value in the specified range
                    hh = np.argmax(signal['lGyro'][hit_range]) + hit_offset  # Pre-heuristic hit

                    # Check if the offset is within bounds
                    if hh + round(1.4 * self.win_sz) <= signal_length:
                        prediction = 1  # If post-heuristic condition, new phase
                    else:
                        prediction = 0  # If not, old phase
                elif self.phases[path['phases'][-1]] == "TS":  
                    # Post-heuristic process: find the index of the minimum value in the specified range
                    hh = np.argmin(signal['lGyro'][hit_range]) + hit_offset  # Pre-heuristic hit location
                    hit_height = signal['lGyro'][hh]  # Minimum value in the range

                    # Check if any value after hh meets the post-heuristic condition
                    if np.any(signal['lGyro'][hh + 1 : signal_length] >= 0.62 * hit_height):
                        prediction = 2  # If post-heuristic condition, new phase
                    else:
                        prediction = 1  # If not, old phase
                elif self.phases[path['phases'][-1]] == "PSW": 
                    # Post-heuristic process: first peak after zero crossing
                    # Find zero crossings in the specified range
                    hh = np.where(np.diff(np.sign(signal['lGyro'][hit_range])) > 0)[0] + hit_offset
                    hh = hh[0]  # Take the first zero crossing

                    # Default prediction
                    prediction = 2

                    # Find unique indices in the peak search range
                    uniqueIdcs = np.unique(signal['lGyro'][hh + 1 : signal_length - 1], return_index=True)[1]
                    uniqueIdcs = uniqueIdcs + hh + 1  # Adjust indices to account for the offset

                    # Check for peaks in the unique indices
                    for j in uniqueIdcs:
                        if signal['lGyro'][j] > signal['lGyro'][j - 1] and signal['lGyro'][j] > signal['lGyro'][j + 1]:  # Peak condition
                            prediction = 3  # If post-heuristic condition, new phase
                            break
                elif self.phases[path['phases'][-1]] == "ISW": 
                    # Post-heuristic process: find the index of the maximum value in the specified range
                    hh = np.argmax(signal['rGyro'][hit_range]) + hit_offset  # Pre-heuristic hit

                    # Check if the offset is within bounds
                    if hh + round(1.4 * self.win_sz) <= signal_length:
                        prediction = 4  # If post-heuristic condition, new phase
                    else:
                        prediction = 3  # If not, old phase
                elif self.phases[path['phases'][-1]] == "MSW":
                    # Post-heuristic process: offset after zero crossing
                    zero_crossings = np.where(np.diff(np.sign(signal['rGyro'][hit_range])) < 0)[0] + hit_offset  
                    hh = zero_crossings[0]  # Pre-heuristic hit
                    
                    if hh + 3 * self.win_sz <= signal_length:  # Column index adjusted
                        prediction = 5  # If post-heuristic, new phase
                    else:
                        prediction = 4  # If not, old phase
                elif self.phases[path['phases'][-1]] == "TSW":
                    # Post-heuristic process: a predetermined percentage of peak value plus offset
                    hit_height, hh = np.min(signal['rGyro'][hit_range]), np.argmin(signal['rGyro'][hit_range])
                    hh += hit_offset  # Adjust hit index with offset

                    TSW_true = np.where(signal['rGyro'][hh:signal_length] >= 0.836*hit_height)[0]  
                    TSW_true = TSW_true + hh + round(0.5 * self.win_sz)  # Adjust indices

                    if TSW_true.size > 0 and TSW_true[0] <= len(signal['rGyro']):
                        prediction = 6  # If post-heuristic, new phase
                    else:
                        prediction = 5  # If not, old phase
                else:
                    prediction = path['phases'][-1]

            """Apply Anchoring : Update the Signal Anchors if Necessary"""
            # Search for new anchors
            if confidence >= 0.5:
                # If the path confidence is high, and we have at least two different path anchor candidates
                anchor_cand_r_locs = path['windows'][np.where(path['phases'][1:] == 0)[0] + 1]
                anchor_cand_l_locs = path['windows'][np.where(path['phases'][1:] == l_anchor_phase)[0] + 1]

                if len(anchor_cand_r_locs) >= 1 and len(anchor_cand_l_locs) >= 1:
                    # Set up the new anchors
                    self.anchor_status[np.max(anchor_cand_r_locs)][0] = True  # New right anchor
                    self.anchor_status[np.max(anchor_cand_l_locs)][1] = True  # New left anchor

                    # Partially reset the array starting with minimum location anchor
                    new_start_idx = min(np.max(anchor_cand_r_locs), np.max(anchor_cand_l_locs))  # New analysis window start

                    self.feedback = self.feedback[new_start_idx * self.win_sz :]  
                    new_likelihood_np = (np.array(self.log_likelihood)[new_start_idx:,:] / np.array(self.log_likelihood)[np.newaxis, new_start_idx - 1, :])
                    self.log_likelihood = [row.squeeze() for row in np.split(new_likelihood_np, new_likelihood_np.shape[0])]
                    self.candidacy = self.candidacy[:, new_start_idx:]
                    self.anchor_status = self.anchor_status[new_start_idx:] 
            
            if self._operation_mode == 'ANN':
                return np.argmax(pure_ann_output), np.max(pure_ann_output)
            else:
                return prediction, confidence
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

class lstm_inf:
    """

    This class classifies gait phases using the LSTM Inference network.
    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        model (tensorflow object): Tensorflow model.
        w (int): Window size.
        s (int): Sequence length.
        n_act (int): Number of activation units.
        modalities (str): Sensory modalities.

    Developer/s:
        Samer A. Mohamed.
        
    """ 
           
    def __init__(self, PrModel, PrModel_meta):
        """

        Class constructor: initializes class parameters.
    
        Args:
            PrModel (str): LSTM tensorflow model path.
            PrModel_meta (str): Model metadata path.
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
            Error: Model metadata path is not a directory or does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """

        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            elif not os.path.exists(PrModel_meta):
                raise FileNotFoundError(f"The directory '{PrModel_meta}' does not exist.")
            else:
                # Define class attributes
                self.model = tf.keras.models.load_model(PrModel, safe_mode=False) # Tensorflow model
                with open(PrModel_meta, 'r') as file:
                    loaded_param = json.load(file) # Load classifier parameters
                self.w = loaded_param['win_sz'] # Window size
                self.s = loaded_param['seq_sz'] # Sequence size
                self.n_act = loaded_param['n_act'] # Number of activations
                self.modalities = loaded_param['modalities'] # Sensor modalities
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.w (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.w
    
    def get_seq_sz(self):
        """

        Get sequence size: returns the model's expected sequence size.

        Args: 
            N/A.
            
        Returns:
            self.s (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.s
    
    def get_modalities(self):
        """

        Get sensor modalities: returns the model's sensor modalities.

        Args: 
            N/A.
            
        Returns:
            self.modalities (str): Sensor modalities.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.modalities
    
    def shut_down(self):
        """

        Sequential prediction reset function: resets the sequential memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        # Reset online belief variables
        self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
    
        return None
    
    def predict(self, signal):
        """

        LSTM prediction function: predicts the gait phase value based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            p (int): Phase prediction.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        try:
            """
            Pattern Recognition
            """
            # Chech if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.w:
                raise ValueError(f"Signal length is not equal to the window size which is {self.w}.")
            
            # Initialize the belief arrays, if not already defined
            if not hasattr(self, 'win_seq'):
                # Online belief variables initialization
                self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
                
            # Append new window to the sequence of windows
            self.win_seq = np.concatenate((self.win_seq, signal[self.modalities].values), axis=0) \
                if self.win_seq.shape[0] < self.s * self.w \
                else np.concatenate((self.win_seq[-(self.s-1)*self.w:,:], \
                                           signal[self.modalities].values), axis=0) 
            if self.win_seq.shape[0] < self.s * self.w:
                p = None; prob = None
            else:
                pred = self.model.predict_on_batch([np.reshape(self.win_seq, \
                           (-1, self.s, self.w*len(self.modalities))), np.zeros((1, self.n_act)), \
                                                            np.zeros((1, self.n_act))])
                p = np.argmax(pred); prob = np.max(pred) # Predicted phase and prediction probability, respectively
            
            return p, prob
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            

class cnn_lstm_inf:
    """

    This class classifies gait phases using the CNN-LSTM Inference network.

    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        model (tensorflow object): Tensorflow model.
        w (int): Window size.
        s (int): Sequence length.
        n_act (int): Number of activation units.
        modalities (str): Sensory modalities.

    Developer/s:
        Samer A. Mohamed.

    """ 

    def __init__(self, PrModel, PrModel_meta):
        """

        Class constructor: initializes class parameters.
    
        Args:
            PrModel (str): CNN-LSTM tensorflow model path.
            PrModel_meta (str): Model metadata path.
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
            Error: Model metadata path is not a directory or does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """
        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            elif not os.path.exists(PrModel_meta):
                raise FileNotFoundError(f"The directory '{PrModel_meta}' does not exist.")
            else:
                # Define class attributes
                self.model = tf.keras.models.load_model(PrModel, safe_mode=False) # Tensorflow model
                with open(PrModel_meta, 'r') as file:
                    loaded_param = json.load(file) # Load classifier parameters
                self.w = loaded_param['win_sz'] # Window size
                self.s = loaded_param['seq_sz'] # Sequence size
                self.n_act = loaded_param['n_act'] # Number of activations
                self.modalities = loaded_param['modalities'] # Sensor modalities
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.w (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.w
    
    def get_seq_sz(self):
        """

        Get sequence size: returns the model's expected sequence size.

        Args: 
            N/A.
            
        Returns:
            self.s (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.s
    
    def get_modalities(self):
        """

        Get sensor modalities: returns the model's sensor modalities.

        Args: 
            N/A.
            
        Returns:
            self.modalities (str): Sensor modalities.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.modalities
    
    def shut_down(self):
        """

        Sequential prediction reset function: resets the sequential memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        # Reset online belief variables
        self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
    
        return None
    
    def predict(self, signal):
        """

        CNN-LSTM prediction function: predicts the gait phase
            value based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            p (int): Phase prediction.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        
        try:
            """
            Pattern Recognition
            """
            # Chech if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.w:
                raise ValueError(f"Signal length is not equal to the window size which is {self.w}.")
            
            # Initialize the belief arrays, if not already defined
            if not hasattr(self, 'win_seq'):
                # Online belief variables initialization
                self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
                
            # Append new window to the sequence of windows
            self.win_seq = np.concatenate((self.win_seq, signal[self.modalities].values), axis=0) \
                if self.win_seq.shape[0] < self.s * self.w \
                else np.concatenate((self.win_seq[-(self.s-1)*self.w:,:], \
                                           signal[self.modalities].values), axis=0) 
            
            # Predict phase using input features
            if self.win_seq.shape[0] < self.s * self.w:
                p = None; prob = None
            else:
                X = np.reshape(self.win_seq, (-1, self.s, self.w, len(self.modalities), 1)).transpose(0, 1, 3, 2, 4) 
                pred = self.model.predict_on_batch([X, np.zeros((1, self.n_act)), np.zeros((1, self.n_act))])
                p = np.argmax(pred); prob = np.max(pred) # Prediction phase and its confidence level
            
            return p, prob
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

class convGRU_inf:
    """

    This class classifies gait phases using the convGRU Inference network.

    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        model (tensorflow object): Tensorflow model.
        w (int): Window size.
        s (int): Sequence length.
        n_act (int): Number of activation units.
        modalities (str): Sensory modalities.

    Developer/s:
        Samer A. Mohamed.

    """ 

    def __init__(self, PrModel, PrModel_meta):
        """

        Class constructor: initializes class parameters.
    
        Args:
            PrModel (str): convGRU tensorflow model path.
            PrModel_meta (str): Model metadata path.
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
            Error: Model metadata path is not a directory or does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """
        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            elif not os.path.exists(PrModel_meta):
                raise FileNotFoundError(f"The directory '{PrModel_meta}' does not exist.")
            else:
                # Define class attributes
                self.model = tf.keras.models.load_model(PrModel, safe_mode=False) # Tensorflow model
                with open(PrModel_meta, 'r') as file:
                    loaded_param = json.load(file) # Load classifier parameters
                self.w = loaded_param['win_sz'] # Window size
                self.s = loaded_param['seq_sz'] # Sequence size
                self.n_act = loaded_param['n_act'] # Number of activations
                self.modalities = loaded_param['modalities'] # Sensor modalities
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.w (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.w
    
    def get_seq_sz(self):
        """

        Get sequence size: returns the model's expected sequence size.

        Args: 
            N/A.
            
        Returns:
            self.s (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.s
    
    def get_modalities(self):
        """

        Get sensor modalities: returns the model's sensor modalities.

        Args: 
            N/A.
            
        Returns:
            self.modalities (str): Sensor modalities.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.modalities
    
    def shut_down(self):
        """

        Sequential prediction reset function: resets the sequential memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        # Reset online belief variables
        self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
    
        return None
    
    def predict(self, signal):
        """

        convGRU prediction function: predicts the gait phase
            value based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            p (int): Phase prediction.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        
        try:
            """
            Pattern Recognition
            """
            # Chech if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.w:
                raise ValueError(f"Signal length is not equal to the window size which is {self.w}.")
            
            # Initialize the belief arrays, if not already defined
            if not hasattr(self, 'win_seq'):
                # Online belief variables initialization
                self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
                
            # Append new window to the sequence of windows
            self.win_seq = np.concatenate((self.win_seq, signal[self.modalities].values), axis=0) \
                if self.win_seq.shape[0] < self.s * self.w \
                else np.concatenate((self.win_seq[-(self.s-1)*self.w:,:], \
                                           signal[self.modalities].values), axis=0) 
            if self.win_seq.shape[0] < self.s * self.w:
                p = None; prob = None
            else:
                pred = self.model.predict_on_batch(np.reshape(self.win_seq, \
                           (-1, self.s, self.w, len(self.modalities))))
                p = np.argmax(pred); prob = np.max(pred) # Predicted phase and prediction probability, respectively
            
            return p, prob
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

class gnn_inf:
    """

    This class classifies gait phases using the ST_GCN Inference network.

    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        model (tensorflow object): Tensorflow model.
        w (int): Window size.
        s (int): Sequence length.
        modalities (str): Sensory modalities.

    Developer/s:
        Samer A. Mohamed.

    """ 

    def __init__(self, PrModel, PrModel_meta):
        """

        Class constructor: initializes class parameters.
    
        Args:
            PrModel (str): gnn tensorflow model path.
            PrModel_meta (str): Model metadata path.
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
            Error: Model metadata path is not a directory or does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """
        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            elif not os.path.exists(PrModel_meta):
                raise FileNotFoundError(f"The directory '{PrModel_meta}' does not exist.")
            else:
                # Define class attributes
                self.model = tf.keras.models.load_model(PrModel, safe_mode=False) # Tensorflow model
                with open(PrModel_meta, 'r') as file:
                    loaded_param = json.load(file) # Load classifier parameters
                self.w = loaded_param['win_sz'] # Window size
                self.s = loaded_param['seq_sz'] # Sequence size
                self.modalities = loaded_param['modalities'] # Sensor modalities
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.w (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.w
    
    def get_seq_sz(self):
        """

        Get sequence size: returns the model's expected sequence size.

        Args: 
            N/A.
            
        Returns:
            self.s (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.s
    
    def get_modalities(self):
        """

        Get sensor modalities: returns the model's sensor modalities.

        Args: 
            N/A.
            
        Returns:
            self.modalities (str): Sensor modalities.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.modalities
    
    def shut_down(self):
        """

        Sequential prediction reset function: resets the sequential memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        # Reset online belief variables
        self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
    
        return None
    
    def predict(self, signal):
        """

        gnn prediction function: predicts the gait phase
            value based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            p (int): Phase prediction.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        
        try:
            """
            Pattern Recognition
            """
            # Chech if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.w:
                raise ValueError(f"Signal length is not equal to the window size which is {self.w}.")
            
            # Initialize the belief arrays, if not already defined
            if not hasattr(self, 'win_seq'):
                # Online belief variables initialization
                self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
                
            # Append new window to the sequence of windows
            self.win_seq = np.concatenate((self.win_seq, signal[self.modalities].values), axis=0) \
                if self.win_seq.shape[0] < self.s * self.w \
                else np.concatenate((self.win_seq[-(self.s-1)*self.w:,:], \
                                           signal[self.modalities].values), axis=0) 
            if self.win_seq.shape[0] < self.s * self.w:
                p = None; prob = None
            else:
                pred = self.model.predict_on_batch(np.reshape(self.win_seq, \
                           (-1, self.s, self.w, len(self.modalities))))
                p = np.argmax(pred); prob = np.max(pred) # Predicted phase and prediction probability, respectively
            
            return p, prob
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

class transformer_inf:
    """

    This class classifies gait phases using the transformer Inference network.

    The class contains methods that set the desired pattern recognition module and
    loads model parameters. The class also contains a prediction method for input 
    windows of sensory readings.

    Attributes:
        model (tensorflow object): Tensorflow model.
        w (int): Window size.
        s (int): Sequence length.
        modalities (str): Sensory modalities.

    Developer/s:
        Samer A. Mohamed.

    """ 

    def __init__(self, PrModel, PrModel_meta):
        """

        Class constructor: initializes class parameters.
    
        Args:
            PrModel (str): gnn tensorflow model path.
            PrModel_meta (str): Model metadata path.
    
        Returns:
            N/A.

        Raises:
            Error: Model path does not exist.
            Error: Model metadata path is not a directory or does not exist.
    
        Developer/s:
            Samer A. Mohamed.

        """
        try:
            if not os.path.exists(PrModel):
                raise FileNotFoundError(f"The directory '{PrModel}' does not exist.")
            elif not os.path.exists(PrModel_meta):
                raise FileNotFoundError(f"The directory '{PrModel_meta}' does not exist.")
            else:
                # Define class attributes
                self.model = tf.keras.models.load_model(PrModel, safe_mode=False, compile=False) # Tensorflow model
                with open(PrModel_meta, 'r') as file:
                    loaded_param = json.load(file) # Load classifier parameters
                self.w = loaded_param['win_sz'] # Window size
                self.s = loaded_param['seq_sz'] # Sequence size
                self.modalities = loaded_param['modalities'] # Sensor modalities
        except (FileNotFoundError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def get_win_sz(self):
        """

        Get window size: returns the model's expected input window size.

        Args: 
            N/A.
            
        Returns:
            self.w (int): Window size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.  

        """

        return self.w
    
    def get_seq_sz(self):
        """

        Get sequence size: returns the model's expected sequence size.

        Args: 
            N/A.
            
        Returns:
            self.s (int): Sequence size.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.s
    
    def get_modalities(self):
        """

        Get sensor modalities: returns the model's sensor modalities.

        Args: 
            N/A.
            
        Returns:
            self.modalities (str): Sensor modalities.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        return self.modalities
    
    def shut_down(self):
        """

        Sequential prediction reset function: resets the sequential memory of 
            the inference object.

        Args: 
            N/A.
            
        Returns:
            N/A.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed. 

        """

        # Reset online belief variables
        self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
    
        return None
    
    def predict(self, signal):
        """

        transformer prediction function: predicts the gait phase
            value based on input window.
            
        Args:
            signal (pandas frame): Data frame of readings from the IMUs. 
            
        Returns:
            p (int): Phase prediction.
            
        Raises:
            Error: Input window is not a pandas data frame of length not equal 
                to window size.

        Developer/s:
            Samer A. Mohamed.  

        """
        
        try:
            """
            Pattern Recognition
            """
            # Chech if input arguments are sound
            if not isinstance(signal, pd.DataFrame):
                raise TypeError("Check argument datatypes: signal must be a pandas data frame.")
            if signal.shape[0] != self.w:
                raise ValueError(f"Signal length is not equal to the window size which is {self.w}.")
            
            # Initialize the belief arrays, if not already defined
            if not hasattr(self, 'win_seq'):
                # Online belief variables initialization
                self.win_seq = np.empty((0, len(self.modalities))) # Sequence of windows
                
            # Append new window to the sequence of windows
            self.win_seq = np.concatenate((self.win_seq, signal[self.modalities].values), axis=0) \
                if self.win_seq.shape[0] < self.s * self.w \
                else np.concatenate((self.win_seq[-(self.s-1)*self.w:,:], \
                                           signal[self.modalities].values), axis=0) 
            if self.win_seq.shape[0] < self.s * self.w:
                p = None; prob = None
            else:
                pred = self.model.predict_on_batch(np.reshape(self.win_seq, \
                           (-1, self.s, self.w, len(self.modalities))))
                p = np.argmax(pred); prob = np.max(pred) # Predicted phase and prediction probability, respectively
            
            return p, prob
        except (TypeError, ValueError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

class BMinf:
    """

    This class classifies gait phases by loading a pretrained model and its 
    prameters. The sensory information is fed to the model in a sequential fashion
    instead of an all-at-once feeding. This is done to mimic the real-time experience
    in which a stream of sensory data is fed to the model window-by-window. Performance 
    metrics are then displayed for one of 6 approaches: lstm, cnn-lstm hybrid, convGRU,
    gnn, transformer, and PHRASE. The stream is obtained from a dataset directory containing 
    CSV files.
    
    The CSV files must start with a code name "ABXXX" followed by a circuit number and
    must contain the following modality headers: ["Right_Shank_Ax","Right_Shank_Az",
    "Right_Shank_Gy","Left_Shank_Ax","Left_Shank_Az","Left_Shank_Gy"] (A: accelerometer, 
    G: gyroscope). Beneath the headers lie sensory data streams from 2 IMUs attached to 
    both shanks of a test subject. The CSV files must also contain "Mode" and "phase" 
    columns, which refer to 'walking activity' and 'gait phase', respectively.

    The dataset has to have a metadata file that mentions the sensory sampling 
    rate and gait phases considered (check the metadata file for BATH_inf in 
    "resources" folder).
    
    The model parameters and metadata must be located in a directory, from which
    the parameters and metadata can be loaded.
    
    The locations and types of sensors are outlined in the following publication:
    https://ieeexplore.ieee.org/abstract/document/10650106
    
    The description of 'Mode' and 'phase' code numbers can be found in:
    https://doi.org/10.15125/BATH-01425

    Attributes:
        ds_path (str): Path to dataset csv files.
        infMode (str): Inference mode.
        model_dir (str): Model parameters path.
        sub_code (str): Test subject code.
        inf_method (str): Class inference method.
        classes (list): Phase labels of interest.
        ds_phases (list): List of phases acknowledged by the inference dataset's metadata.
        freq (float): Sampling rate of the inference dataset.
        PHRASE_mode (str): Operation mode of PHRASE (if selected)
        severe (boolean): Severity indicator in case of impairment

    Developer/s:
        Samer A. Mohamed.

    """
    def __init__(self, Mparam, infMode = "test", classes=['LR', 'MST', 'TS', 'PSW', 'SW'], PHRASE_mode='full', severe = False):
        """

        Class constructor: initializes class parameters.
    
        Args:
            Mparam (str): Model loading directory.
            infMode (str): Inference mode (train, validation or test; default: test)
            classes (list): List of strings representing class labels.
            PHRASE_mode (str): Operation mode of PHRASE (if selected)
            severe (boolean): Severity indicator in case of impairment
    
        Returns:
            N/A.

        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """
        # Initialize class attributes 
        self.sub_code = None
        self.ds_path = None
        self.inf_method = None
        self.classes = classes # Walking phases
        self.PHRASE_mode = PHRASE_mode
        self.severe = severe
        
        # Define class attributes
        self.set_path(Mparam=Mparam) # Set class paths

        # Check class definition is standard
        if classes != ['LR', 'MST', 'TS', 'PSW', 'SW'] and \
            classes != ['LR', 'MST', 'TS', 'PSW', 'ISW', 'MSW', 'TSW']:
            raise ValueError("Desired phase definitions are not standard")

        # Check if inference model is correct
        if infMode not in ["train", "valid", "test"]:
            raise ValueError("Inference mode can only be train, valid, or test.")
        else:
            self.infMode = infMode
        
    def set_path(self, Dset=None, Mparam=None):
        """

        Path assignment: sets the class paths.
        
        Args:
            Dset (str): Dataset path.
            Mparam (str): Model parameters path.

        Returns:
            N/A.
            
        Raises:
            Error: Path does not exist or is not a valid directory.
            
        Developer/s:
            Samer A. Mohamed.

        """
        # Group the paths
        Paths = [item for item in [Dset, Mparam] if item is not None]
        try:
            # Check that paths are valid
            for i in range(len(Paths)):
                if not os.path.exists(Paths[i]):
                    raise FileNotFoundError(f"The directory '{Paths[i]}' does not exist.")
                elif not os.path.isdir(Paths[i]):
                    raise NotADirectoryError(f"'{Paths[i]}' is not a directory.")                  
            # Update paths
            self.ds_path = Dset if Dset != None else self.ds_path 
            self.model_dir = Mparam if Mparam != None else self.model_dir

            # Set dataset frequency property
            if self.ds_path:
                with open(self.ds_path + "/metadata.json", 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.freq = metadata["signal_parameters"]["sampling_frequency"]
                self.ds_phases = metadata["gait_phases"]["phase_names"]
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"ERROR: {str(e)}")
            sys.exit()

    def set_test_sub(self, u=None):
        """

        Test subject assignment: sets the test subject code name.
        
        Args:
            u (str): Code name. 

        Returns:
            N/A.
            
        Raises:
            Error: Dataset path doesn't exist.
            Error: Test subject number doesn't exist.
            Error: Test subject code must be a string.
            
        Developer/s:
            Samer A. Mohamed.

        """
        # Set unseen subject
        if type(u) == str:
            try:
                # Check if the dataset path has been set
                if self.ds_path:
                    # Locate CSV files
                    csv_file_names = sorted([f for f in os.listdir(self.ds_path) if f.endswith('.csv')]) 
                    if any(u in element for element in csv_file_names):
                        self.sub_code = str(u) # Update unseen subject code name
                    else:
                        raise ValueError(f"The code number '{u}' does not exist in the dataset directory," \
                                         " please select a valid subject from the dataset or check of the dataset is empty.")
                else:
                    raise NameError("The dataset path is not defined, please use the 'set_path' method" \
                                    "to set the dataset path.")
            except (ValueError, NameError) as e:
                print(f"ERROR: {str(e)}")
                sys.exit() 
        else:
            # Display an error: empty dataset
            print("ERROR: Test subject code must be a string.")
            sys.exit()
            
    def get_dataset(self):
        """

        Dataset path return: gets the path of the assigned dataset.
        
        Args:
            N/A. 

        Returns:
            self.ds_path: dataset path assigned to the class.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.

        """
        return self.ds_path
    
    def get_sub_code(self):
        """

        Test subject code return: gets the name of the test subject.
        
        Args:
            N/A. 

        Returns:
            self.sub_code: name of the test subject.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.

        """
        return self.sub_code
    
    def _set_inf_method(self, m):
        """

        Method setting: sets the classification approach.
        
        Args:
            m (str): Method (lstm, cnn-lstm, etc.). 

        Returns:
            N/A.
            
        Raises:
            Error: Incorrect method.
            
        Developer/s:
            Samer A. Mohamed.

        """
        # Check that input arguments are valid
        try:
            if m not in ["lstm", "cnn-lstm", "convGRU", "gnn", "transformer", "phrase"]:
                raise ValueError(f"The method '{m}' does not exist. Please select from" + \
                      ": 'lstm', 'cnn-lstm', 'convGRU', 'gnn', 'transformer', and 'phrase'.")
            else:
                self.inf_method = m
        except ValueError as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
    def _DSpcut(self, ct_list):
        """

        Dataset phase cut function: cuts the raw CSV file data into separate
            dataframes, each containing a continuous gait activity instance.
    
        Args:
            ct_list (list): List of circuit CSV file names.
    
        Returns:
            df_list (list): List of gait data frames.

        Raises:
            Error: trial/circuit does not follow standard format.
    
        Developer/s:
            Samer A. Mohamed.
            
        """
        # Loop over circuit lists
        df_list = []
        for i in range(len(ct_list)):
            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(self.ds_path+"/"+ct_list[i]) # Data frame
                df = df[df["Mode"]==1]
            except ValueError:
                print(f"ERROR: circuit file '{ct_list[i]}' does not follow standard format.")
                sys.exit()
            
            # Break the CSV file into separate activity cycle data frames
            if df.shape[0] > 0:
                idx = np.append(np.append(0,np.where(np.diff(df.index)!=1)[0]+1),df.index.shape[0]) # Separation indices
                for k in range(idx.shape[0]-1):
                    df_list.append(df[idx[k]:idx[k+1]]) # Append new activity cycle
                
        return df_list
        
    def _compute_metrics(self, true_labels, predicted_labels):
        """

        Performance metrics: calculate and display.
        
        Args:
            true_labels (nd.array): Ground truth labels. 
            predicted_labels (nd.array): Predicted labels from network.
            fold (str): Unseen subject fold code.
        
        Returns:
            performance (tuple): Scores for the fold.
        
        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.  

        """ 
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, 
            average=None  # ← This gives per-class values
        )

        # Display accuracy
        overall_accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"  overall_accuracy: {overall_accuracy*100.0}%")

        # Display precision for each class
        print("   Precision for Each Class:")
        for i in range(len(precision)):
            print(f"  {self.classes[i]}: {precision[i]}")

        # Display recall for each class
        print("   Recall for Each Class:")
        for i in range(len(recall)):
            print(f"  {self.classes[i]}: {recall[i]}")

        # Display F1 score for each class
        print("    F1 score for Each Class:")
        for i in range(len(f1)):
            print(f"  {self.classes[i]}: {f1[i]}")

        # Compute specificity
        print("    Specificity for Each Class:")
        cm = confusion_matrix(true_labels, predicted_labels)
        specificity = np.zeros(cm.shape[0],)
        for i in range(cm.shape[0]):
            tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
            fp = cm[:,i].sum() - cm[i,i]
            specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"  {self.classes[i]}: {specificity[i]}")

        # Plot confusion matrix
        """
        plt.figure(figsize=(cm.shape[0], cm.shape[1]))
        cm_plt = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes, \
                    annot_kws={'size': 16, 'weight': 'bold'})
        cbar = cm_plt.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.xlabel('Predicted Labels', fontsize=16, fontweight='bold')
        plt.ylabel('True Labels', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.title('Confusion Matrix: ' + self.sub_code + ', ' + self.inf_method + ', ' + 'W', \
                  fontsize=16, fontweight='bold')
        plt.show()
        """
            
        return overall_accuracy, precision, recall, f1, specificity

    def get_inf_metrics(self, model):
        """

        Inference procedure: applied inference on data from a specific subject or fold.
        
        Args:
            model: Inference model. 
        
        Returns:
            performance (tuple): Scores for the fold.
        
        Raises:
            Error: Test subject code unspecified.
            Error: model and metadata are not matching.
            Error: Insufficient inference data.
    
        Developer/s:
            Samer A. Mohamed.  

        """ 
        # Set model
        self._set_inf_method(model)

        # Determine the most appropriate model to use given the test subject code
        if self.sub_code:
            if self.inf_method != 'phrase':
                # Sort all model and metadata files in the model directory
                meta_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.json') and str(round(self.freq)) in f and f.startswith(self.inf_method)])
                model_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.keras') and str(round(self.freq)) in f and f.startswith(self.inf_method)])
                if meta_files == [] or model_files == []:
                    raise FileNotFoundError("Empty model directory!!")

                # Select the model and metadata that match the test code as unseen subject in internal dataset during training
                # If the test subject doesn't exist at all as unseen subject in the directory (external dataset), pick the first available model and metadata
                meta_of_choice = os.path.join(self.model_dir, next((f for f in meta_files if self.sub_code in f), meta_files[0]))
                model_of_choice = os.path.join(self.model_dir, next((f for f in model_files if self.sub_code in f), model_files[0]))
                if os.path.splitext(meta_of_choice)[0] != os.path.splitext(model_of_choice)[0]:
                    raise ValueError("Model and metadata are not matching, check the model directory and make sure each model has associated metadata")
                if self.sub_code not in model_of_choice:
                    warnings.warn("Test code doesn't exist in model directory. Make sure that the test code belongs to an external dataset.")

                # Check if the definitions of phases are compatible
                with open(meta_of_choice, 'r', encoding='utf-8') as f:
                    model_metadata = json.load(f)
            else:
                # Sort all model files in the model directory
                model_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.json') and f.startswith(self.inf_method)])
                if model_files == []:
                    raise FileNotFoundError("Empty model directory!!")

                # Select the model that matches the test code as unseen subject in internal dataset during training
                # If the test subject doesn't exist at all as unseen subject in the directory (external dataset), pick the first available model and metadata
                model_of_choice = os.path.join(self.model_dir, next((f for f in model_files if self.sub_code in f), model_files[0]))
                if self.sub_code not in model_of_choice:
                    warnings.warn("Test code doesn't exist in model directory. Make sure that the test code belongs to an external dataset.")

                # Check if the definitions of phases are compatible
                with open(model_of_choice, 'r', encoding='utf-8') as f:
                    model_metadata = json.load(f)
            
            if (not (model_metadata["phases"] == self.ds_phases)) and \
                (not(model_metadata["phases"] == ["LR", "MST", "TS", "PSW", "ISW", "MSW", "TSW"] and \
                self.ds_phases == ["LR", "MST", "TS", "PSW", "SW"])):
                raise ValueError("Phase definitions of model and inference dataset are incompatible.")
            else:
                if len(self.ds_phases) < len(self.classes):
                    warnings.warn("The dataset phase definition is less specific than the desired phase classes.")
                    warnings.warn("Using dataset phase definitions instead...")
                    self.classes = self.ds_phases
        else:
            raise ValueError("Test subject code have not been specified, please specifify using set_test_sub()")
            
        # Create inference folds
        print("LOG: Creating folds....\n")   
        if self.infMode == "test":
            inf_list = sorted([f for f in os.listdir(self.ds_path) if f.endswith('.csv') and self.sub_code in f])  # Locate CSV files associated with the test code
        else:
            full_list = sorted([f for f in os.listdir(self.ds_path) if f.endswith('.csv') and self.sub_code not in f])  # Locate CSV files NOT associated with the test code
            unique_codes = {s[:5] for s in full_list}  # Using set comprehension
            inf_list = []
            for unique_code in unique_codes:
                # Get files starting with this prefix
                sub_subset = [f for f in full_list if f.startswith(unique_code)]
                split_idx = int(0.6 * len(sub_subset))
                if self.infMode == "train":
                    # Take first 60% for training
                    inf_list.extend(sub_subset[:split_idx])  # Use extend, not append
                else:  
                    # Take last 40% for validation
                    inf_list.extend(sub_subset[split_idx:])
            
        # Cut the CSV files into separate continuous gait cycles
        inf_cut = self._DSpcut(inf_list)
        print(f"LOG: '{self.sub_code}' files in dataset directory were segmented successfully!\n")
            
        # Create the classifier inference object
        if self.inf_method == 'phrase':
            self.classifier = phrase_inf(fs = self.freq, PrModel=model_of_choice, mode=self.PHRASE_mode, severe = self.severe)
        elif self.inf_method == 'lstm':
            self.classifier = lstm_inf(PrModel=model_of_choice, PrModel_meta=meta_of_choice) 
        elif self.inf_method == 'cnn-lstm':
            self.classifier = cnn_lstm_inf(PrModel=model_of_choice, PrModel_meta=meta_of_choice)  
        elif self.inf_method == 'convGRU':
            self.classifier = convGRU_inf(PrModel=model_of_choice, PrModel_meta=meta_of_choice)  
        elif self.inf_method == 'gnn':
            self.classifier = gnn_inf(PrModel=model_of_choice, PrModel_meta=meta_of_choice)  
        elif self.inf_method == 'transformer':
            self.classifier = transformer_inf(PrModel=model_of_choice, PrModel_meta=meta_of_choice)  
        else:
            raise ValueError("Inference method has not been specified, please specify using _set_inf_method()")
        self.s = self.classifier.get_seq_sz() # Get the model's sequence size (seq size is not required for PHRASE, but just to unite the evaluation results in terms of dataset size)
        self.w = self.classifier.get_win_sz() # Get the model's window size

        # Loop over the individual testing gait bouts
        true_labels = np.array([]); predicted_labels = np.array([])
        exec_time = np.array([]) # Average inference time "initialized to empty"
        td = np.array([]) # Time delay between true and predicted transitions

        # Loop over walking bouts
        for bout in range(len(inf_cut)):
            # Initialize label vectors for current bout
            true_labels_bout = np.array([])
            predicted_labels_bout = np.array([])

            # Loop over bout windows
            for i in range(self.w, inf_cut[bout].shape[0], self.w):
                # Cut readings widnow
                signal = inf_cut[bout].iloc[i-self.w:i] 
                
                # Start timer
                start_time = time.time()
                
                # Apply inference 
                # make sure classifier output is limited to the number of classes in case it was trained to be more specific
                # e.g., in come models, swing is divided into three more distinct phases
                phase, _ = self.classifier.predict(signal)
                if phase:
                    phase = min(phase, len(self.classes)-1)
                
                # End timer
                end_time = time.time()
                exec_time = np.append(exec_time, end_time - start_time)
                
                # Update label record
                # Wait till at least one whole sequence is present
                if i >= self.s*self.w:
                    predicted_labels_bout = np.append(predicted_labels_bout, phase)
                    true_labels_bout = np.append(true_labels_bout, min(signal['phase'].values[-1], len(self.classes)-1))  # Cap true label in case the dataset's phases are more specific than trainer
            
            # Reset classifier at the end of a complete bout
            self.classifier.shut_down() 
            
            # Time delay calculation
            pred_trans_idx = np.where(np.diff(predicted_labels_bout))[0]+1 #--> Predicted transition instances
            for k in range(len(pred_trans_idx)):
                # Find the limits of the current gait cycle
                limit_start = np.where(np.logical_and(np.append(0,np.diff(true_labels_bout)), true_labels_bout==0))[0] # Gait cycles' starts in bout
                limit_end = limit_start-1
                limit_start = limit_start[limit_start<=pred_trans_idx[k]] # Starts that come before predicted trans.
                if limit_start.shape[0] != 0:
                    limit_start = pred_trans_idx[k]-min(pred_trans_idx[k]-limit_start) # Closest start to predicted trans.
                else:
                    limit_start = 0 # If no criteria are matched, pick the bout start as the gait cycle start
            
                limit_end = limit_end[limit_end>pred_trans_idx[k]] # Starts that come after predicted trans.
                if limit_end.shape[0] != 0:
                    limit_end = min(limit_end-pred_trans_idx[k]) + pred_trans_idx[k] # Closest start to predicted trans.
                else:
                    limit_end = len(true_labels_bout); # If no criteria are matches, pick the bout start as the gait cycle start
                
                # Find the corrsponding true transition & calculate delay
                # (delay unit: windows)
                true_trans_idx = np.where(np.logical_and(true_labels_bout[1:]==predicted_labels_bout[pred_trans_idx[k]], \
                    true_labels_bout[0:-1]==predicted_labels_bout[pred_trans_idx[k]-1]))[0]
                true_trans_idx = true_trans_idx[np.logical_and(true_trans_idx>=limit_start, true_trans_idx < limit_end)]
                if true_trans_idx.shape[0] != 0:
                    true_loc = np.argmin(abs(true_trans_idx-pred_trans_idx[k]))
                    td = np.append(td, pred_trans_idx[k]-true_trans_idx[true_loc]) # Transition prediction delay
                
            # Operational prediction adjustment (neglect one sample delay slippage)
            transitions_idx = np.where(true_labels_bout[:-1] != true_labels_bout[1:])[0] + 1                        # True phase transition indices
            cool_down_idx = np.concatenate((transitions_idx-1, transitions_idx, transitions_idx+1))                 # Cool down zone
            cool_down_idx = cool_down_idx[np.logical_and(cool_down_idx>=0,cool_down_idx<len(true_labels_bout))]     # Remove indices that lie outside of the array
            true_labels_bout = np.delete(true_labels_bout, cool_down_idx)
            predicted_labels_bout = np.delete(predicted_labels_bout, cool_down_idx)

            # Append the current bout labels to the global array of labels across all bouts
            true_labels = np.concatenate([true_labels, true_labels_bout]) 
            predicted_labels = np.concatenate([predicted_labels, predicted_labels_bout])
        
        # Extract metrics
        if self.infMode == "test":
            print("Testing Metrics:")
        elif self.infMode == "train":
            print("Training Metrics:")
        else:
            print("Validation Metrics:")
        print(f"  Average inference time: {np.mean(exec_time)} +/- {np.std(exec_time)} seconds.")
        overall_accuracy, precision, recall, f1, specificity = self._compute_metrics(true_labels, predicted_labels)
        print(f"  Delay: {np.mean(td*self.w)} +/- {np.std(td*self.w)} Hz.sec.")

        return overall_accuracy, precision, recall, f1, specificity