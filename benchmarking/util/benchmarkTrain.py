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
from tensorflow.keras.layers import Permute, BatchNormalization, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras import backend as K
import numpy as np
import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

class BMtrainer:
    """

    This class is used to train a gait phase classifier using different benchmarks
    on datasets containing CSV files of a particular format. The CSV files must
    start with a code name "ABXXX" followed by a trial number. The CSV files 
    must contain the following modality headers: ["Right_Shank_Ax","Right_Shank_Az",
    "Right_Shank_Gy","Left_Shank_Ax","Left_Shank_Az","Left_Shank_Gy"] 
    (A: accelerometer, G: gyroscope). 

    Beneath the headers lie sensory data streams from 2 IMUs attached to both 
    shanks of a test subject. 
    
    The CSV files must also contain "Mode" and "phase" columns, which refer to 
    'walking activity' and 'gait phase', respectively. The class allows 
    the user to train using LSTM, CNN-LSTM hybrid, convGRU, graph neural networks 
    or transformer methods. 

    The dataset has to have a metadata files that mentions the sensory sampling 
    rate and gait phases considered (check the metadata file for BLISS_inf in 
    "resources" folder).
    
    The locations and types of sensors are outlined in the following publication:
    (S. A. Mohamed and U. Martinez-Hernandez, "Wearable Interface for Real-time Gait 
    Phase Recognition using Sensor Networks," Applied Soft Computing, 2026.)
    
    The description of 'Mode' and 'phase' code numbers can be found in:
    https://doi.org/10.15125/BATH-01425 

    Attributes:
        modalities (list): Sensor modalities in a list of strings.
        phases (str): Gait phases considered during training.
        freq (float): Sensory sampling rate for the raw training dataset.
        target_fs (float): Sensory sampling rate for the expetced testing scenario.
        overlap (int): Overlap between classification windows in the dataset.
        ds_path (str): Path to the training dataset directory.
        trainer (str): Classification method.
        w (int): Window size.
        s (int): Sequence size, relevant only to lstm and cnn-lstm methods.
        name (str): Unseen subject code.
        save_dir (str): The directory in which model parameters are saved after training.
        tmodel (class object): Training model object.

    Developer/s:
        Samer A. Mohamed.

    """

    def __init__(self, path, method="lstm", seq_sz=10, unseen_sub=None, \
                 Mdirectory=None, fs_ratio=1):
        """

        Class constructor: initializes class parameters.
    
        Args:
            path (str): Path to CSV files directory.
            method (str): Training method (default: lstm).
            seq_sz (int): Length of the sequence of windows fed to a network (default: 10).
            unseen_sub (int): Unseen subject code number (optional).
            Mdirectory (str): Model saving directory (optional).
            fs_ratio (float): The ratio between the training sampling rate and the 
                expected testing sampling rate (default: 1).
    
        Returns:
            N/A.

        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        # Define class attributes
        self.modalities = ["Right_Shank_Ax","Right_Shank_Az","Right_Shank_Gy", \
            "Left_Shank_Ax", "Left_Shank_Az","Left_Shank_Gy"]                               # Sensor modalities of interest
        self.set_dataset(path)                                                              # Dataset setting
        self.target_fs = round(self.freq/fs_ratio)                                          # Target testing sampling rate                                                                     
        self.overlap = round(self.target_fs/100.0)                                          # Shift size
        self.set_train_param(method, round(self.target_fs*0.03), seq_sz)                    # Training method + input format
        self.set_unseen_sub(unseen_sub)                                                     # Unseen subject of the fold
        self.sample_ratio = round(fs_ratio)                                                 # Sampling ratio (Fs_train / Fs_test)
        
        # No parameters directory -> default saving directory
        if Mdirectory is None:
            self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints')     
                                                                                            # Model saving directory
        else:
            self.save_dir = Mdirectory
    
    def set_train_param(self, m, w, s=10):
        """

        Method setting: sets the training approach/network type and input frame size.
        
        Args:
            m (str): Method (lstm, cnn-lstm, etc.) 
            w (int): Window size.
            s (int): Sequence size for sequence/temporal models.

        Returns:
            N/A.
            
        Raises:
            Error: Incorrect method.
            Error: Incorrect window size (-ve or non-float).
            Error: Incorrect sequence size (-ve or non-float).
            
        Developer/s:
            Samer A. Mohamed.

        """

        # Check that input arguments are valid
        try:
            if m not in ["lstm", "cnn-lstm", "convGRU", "gnn", "transformer"]:
                raise ValueError(f"The method '{m}' does not exist. Please select from" + \
                      ": 'lstm', 'cnn-lstm', 'convGRU'. 'gnn', and 'transformer'.")
            elif not isinstance(w, int) or w <= 0:
                raise ValueError("Window size must be integer and positive.")
            elif not isinstance(s, int) or s <= 0:
                raise ValueError("Sequence size must be integer and positive.")
            else:
                self.trainer = m; self.w = w; self.s = s
        except ValueError as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
        
    def get_train_param(self):
        """

        Training parameters feedback: gets the training parameters of the 
            class.
        
        Args:
            N/A. 

        Returns:
            self.trainer: Training method assigned to the class.
            self.w: Input window size.
            self.s: Input sequence size for sequence/temporal models.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.

        """

        return self.trainer, self.w, self.s
    
    def set_dataset(self, p):
        """

        Dataset assignment: sets the training dataset path.
        
        Args:
            p (str): Dataset path. 

        Returns:
            N/A.
            
        Raises:
            Error: Dataset path does not exist or is not a valid directory.
            
        Developer/s:
            Samer A. Mohamed.

        """

        # Check that input arguments are valid
        try:
            if not os.path.exists(p):
                raise FileNotFoundError(f"The directory '{p}' does not exist.")
            elif not os.path.isdir(p):
                raise NotADirectoryError(f"'{p}' is not a directory.")
            else:
                self.ds_path = p # Setting dataset path
                with open(self.ds_path + "/metadata.json", 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.freq = metadata["signal_parameters"]["sampling_frequency"]
                self.phases = metadata["gait_phases"]["phase_names"]
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"ERROR: {str(e)}")
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
        
    def set_unseen_sub(self, u):
        """

        Unseen subject assignment: sets the unseen test subject code
            name.
        
        Args:
            u (int): Code name. 

        Returns:
            N/A.
            
        Raises:
            Error: Dataset path doesn't exist.
            Error: Incorrect unseen subject number.
            Error: Empty dataset directory.
            
        Developer/s:
            Samer A. Mohamed.

        """

        # Locate CSV files
        csv_file_names = sorted([f for f in os.listdir(self.ds_path) if f.endswith('.csv')]) 
        
        # Set unseen subject
        if u is not None:
            try:
                # Check if the dataset path has been set
                if hasattr(self, 'ds_path'):
                    if any(int(element[2:5]) == u for element in csv_file_names):
                        self.name = str(u) # Update unseen subject code name
                    else:
                        raise ValueError(f"The code number '{u}' does not exist in the dataset directory," \
                                         " please select a valid subject from the dataset.")
                else:
                    raise NameError("The dataset path is not defined, please use the 'set_dataset' method" \
                                    "to set the dataset path.")
            except (ValueError, NameError) as e:
                print(f"ERROR: {str(e)}")
                sys.exit() 
        elif csv_file_names:
            # Select a random unseen subject
            # Check if the random subject follows the naming convention
            try:
                self.name = random.choice(csv_file_names)[2:5]
            except ValueError:
                print("ERROR: The random unseen subject doesn't follow the standard naming convention." \
                      " Specify your own unseen subject from the dataset to avoid confusion (" \
                      "use 'set_unseen_sub' method explicitly or initialize unseen subject during object creation).")
                sys.exit() 
        else:
            # Display an error: empty dataset
            print("ERROR: The provided dataset directory is empty, please choose a valid directory.")
            sys.exit()
    
    def get_unseen_sub(self):
        """

        Unseen code name return: gets the name of the unseen subject.
        
        Args:
            N/A. 

        Returns:
            self.name: unseen subject code name.
            
        Raises:
            N/A.
            
        Developer/s:
            Samer A. Mohamed.

        """

        return self.name
    
    def _DSpcut(self, ct_list):
        """

        Dataset phase cut function: cuts the raw CSV file data into separate
            dataframes, each containing a continuous gait bout.
    
        Args:
            ct_list (list): List of circuit CSV file names.
    
        Returns:
            df_list (list): List of gait data frames.

        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        # Loop over circuit lists
        df_list = []
        for i in range(len(ct_list)):
            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(self.ds_path+"/"+ct_list[i], usecols=self.modalities+["Mode","phase"]) # Data frame
                df = df[df["Mode"]==1]
            except ValueError:
                print(f"ERROR: circuit file '{ct_list[i]}' does not follow standard format.")
                sys.exit()
            
            # Break the CSV file into separate activity cycle data frames
            if df.shape[0] > 0:
                idx = np.append(np.append(0,np.where(np.diff(df.index)!=1)[0]+1),df.index.shape[0]) # Separation indices
                for k in range(idx.shape[0]-1):
                    df_list.append(df[idx[k]:idx[k+1]:self.sample_ratio]) # Append new activity cycle
                    #df_list.append(df[idx[k]:idx[k+1]:2]) # For 250 Hz online signal
                
        return df_list
    
    def _sliding_window(self, matrix, window_size, overlap):
        """

        Sliding window function: segment the data by sliding a window with a 
            particular stride.
    
        Args:
            matrix (ndarray): Array of sensor readings/labels.
            window_size (int): Size of the sliding window.
            overlap (int): No. of overlapping elements between 
                successive windows.
    
        Returns:
            ** (ndarray): Array of segmented data.

        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        # Calculate the number of steps and strides
        num_steps = (matrix.shape[0] - window_size) // overlap + 1
        strides = overlap * matrix.strides[0]

        # Create a view with the specified window size and overlap
        view_shape = (num_steps, window_size, matrix.shape[1])
        
        return np.lib.stride_tricks.as_strided(matrix, shape=view_shape, strides=(strides,) + matrix.strides)
    
    def _DSpprocess(self, cut):
        """

        Dataset phase preprocess function: preprcoesses the activity cycle 
            dataframes to extract features and labels.
    
        Args:
            cut (list): List of gait data frames.
    
        Returns:
            X (nd.array): Array of processed features.
            y (nd.array): Array of associated labels.

        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        if(self.trainer == "lstm"):
            # Initialize the features and labels
            X = np.reshape(self._sliding_window(cut[0].values[:,:-2], self.s*self.w, self.overlap), \
                           (-1, self.s, self.w*len(self.modalities))) # Features for LSTM
            y = np.reshape(self._sliding_window(cut[0].values[:,-1].reshape(-1,1), self.s*self.w, \
                                                self.overlap),(-1,self.s*self.w)) # Grouping labels
            y = [row[-1] for row in np.int64(y[:,-self.w:])] # Labels
            
            # Loop to fill dataset dictionaries
            for i in range(1, len(cut)):
                    # Extract features
                    X = np.concatenate((X, np.reshape(self._sliding_window(cut[i].values[:,:-2], self.s*self.w, self.overlap), \
                                   (-1, self.s, self.w*len(self.modalities)))), axis=0) # Features for LSTM
                    labels = np.reshape(self._sliding_window(cut[i].values[:,-1].reshape(-1,1), self.s*self.w, \
                                                        self.overlap),(-1,self.s*self.w)) # Grouping labels
                    labels = [row[-1] for row in np.int64(labels[:,-self.w:])] 
                    y = np.concatenate((y,labels), axis=0) # Labels
        elif(self.trainer == "cnn-lstm"):
            # Initialize the features and labels
            X = np.reshape(self._sliding_window(cut[0].values[:,:-2], self.s*self.w, self.overlap), \
                           (-1, self.s, self.w, len(self.modalities), 1)) # Features for LSTM
            y = np.reshape(self._sliding_window(cut[0].values[:,-1].reshape(-1,1), self.s*self.w, \
                                                self.overlap),(-1,self.s*self.w)) # Grouping labels
            y = [row[-1] for row in np.int64(y[:,-self.w:])] # Labels
            
            # Loop to fill dataset dictionaries
            for i in range(1, len(cut)):
                    # Extract features
                    X = np.concatenate((X, np.reshape(self._sliding_window(cut[i].values[:,:-2], self.s*self.w, self.overlap), \
                                   (-1, self.s, self.w, len(self.modalities), 1))), axis=0) # Features for LSTM
                    labels = np.reshape(self._sliding_window(cut[i].values[:,-1].reshape(-1,1), self.s*self.w, \
                                                        self.overlap),(-1,self.s*self.w)) # Grouping labels
                    labels = [row[-1] for row in np.int64(labels[:,-self.w:])] 
                    y = np.concatenate((y,labels), axis=0) # Labels            
            X = X.transpose(0, 1, 3, 2, 4) 
                #--> Transpose to get (no. of exp, no. of seq., no. of sensory channels, no. of samples, no. of CNN channels)
        elif(self.trainer == "convGRU" or self.trainer == "gnn" or self.trainer == "transformer"):
            # Initialize features and labels from first segment
            # Features: all columns except last 2
            raw_features = cut[0].values[:, :-2]  # Shape: (total_samples, modalities)
            
            # Apply sliding window to features
            windowed_features = self._sliding_window(raw_features, self.s * self.w, self.overlap)
            # Shape: (num_windows, s*w, modalities)
            
            # Reshape to create timesteps dimension
            # From (num_windows, s*w, modalities) to (num_windows, s, w, modalities)
            X = windowed_features.reshape(-1, self.s, self.w, len(self.modalities))
            
            # Labels processing
            raw_labels = cut[0].values[:, -1].reshape(-1, 1)  # Last column as labels
            windowed_labels = self._sliding_window(raw_labels, self.s * self.w, self.overlap) # Shape: (num_windows, s*w, 1)
            labels_3d = windowed_labels.reshape(-1, self.s, self.w) # Reshape to access the structure: (num_windows, s, w)
            y = labels_3d[:, -1, -1].astype(np.int64)  # Take the last sample of the entire window as label, Shape: (num_windows,)
            
            # Loop through remaining segments
            for i in range(1, len(cut)):
                # Features for segment i
                raw_features_i = cut[i].values[:, :-2]
                windowed_features_i = self._sliding_window(raw_features_i, self.s * self.w, self.overlap)
                X_i = windowed_features_i.reshape(-1, self.s, self.w, len(self.modalities))
                X = np.concatenate((X, X_i), axis=0)
                
                # Labels for segment i
                raw_labels_i = cut[i].values[:, -1].reshape(-1, 1)
                windowed_labels_i = self._sliding_window(raw_labels_i, self.s * self.w, self.overlap)
                labels_3d_i = windowed_labels_i.reshape(-1, self.s, self.w)
                y_i = labels_3d_i[:, -1, -1].astype(np.int64)
                y = np.concatenate((y, y_i), axis=0)
            
        # Random shuffle
        idx = np.arange(X.shape[0]) #--> Range of array indices
        np.random.shuffle(idx) #--> Shuffle indices
        X = X[idx, :] #--> Shuffled features
        y = y[idx] #--> Shuffled labels
                
        return X, y
                
    def _convert_to_one_hot(self, y):
        """

        Label conversion: converts y into one hot reprsentation.

        Args:
            y (nd.array): An array containing integer values.

        Returns:
            one_hot (nd.array): One-hot representation of y.
            
        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        max_value = max(y)
        min_value = min(y)
        length = len(y)
        one_hot = np.zeros((length, (max_value - min_value + 1)))
        one_hot[np.arange(length), y - min_value] = 1
        
        return one_hot 
    
    def _convert_from_one_hot(self, y):
        """

        Label conversion: converts y back from one hot/probability reprsentation 
            to normal.

        Args:
            y (nd.array): An array containing class probabilities/one hot form.

        Returns: 
            one_hot (nd.array): Normal 1D representation of y.
            
        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """

        y_origin = np.argmax(y, axis = 1)
        return y_origin
    
    def _lstm_model(self, Tx, n_a, n_values, n_o):
        """

        LSTM model: builds the model.
        
        Args:
            Tx (int): Length of the sequence in a data stream.
            n_a (int): The number of activations used in this model.
            n_values (int): Number of input features for a single frame/window.
            n_o (int): Number of output units.
        
        Returns:
            model (keras model): LSTM keras model.
        
        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed. 

        """    

        # Define the input of your model with a shape 
        X = Input(shape=(Tx, n_values))
        
        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        
        # Step 1: Loop
        for t in range(Tx):
            # Step A: Get the input for time "t"
            x = Lambda(lambda x_input: x_input[:, t, :])(X)
            # Step B: Reshape x to be (1, n_values)
            x = Reshape((1, n_values))(x)
            # Step C: Perform one step of the LSTM_cell
            a, _, c = LSTM(n_a, return_state = True)(x, initial_state=[a, c])
            
        # Step 2: Apply densor to the hidden state output of LSTM_Cell after iterating through time
        out = Dense(n_o, activation='softmax')(a)
            
        # Step 3: Create model instance
        model = Model([X, a0, c0], out)
        
        return model
    
    def _cnn_lstm_model(self, Tx, n_a, n_sens, n_frames, nc, n_o):
        """

        CNN-LSTM model: builds the model.
        
        Arguments:
            Tx (int): Length of the sequence in a data stream.
            n_a (int): The number of activations used in this model.
            n_sens (int): Number of sensors.
            n_frames (int): Number of frames/windows.
            nc (int): Number of channels.
            n_o (int): Number of output units.
        
        Returns:
            model (keras model): LSTM keras model.
        
        Raises:
            N/A.
    
        Developer/s:
            Samer A. Mohamed.

        """ 

        # Define the input of your model with a shape 
        X = Input(shape=(Tx, n_sens, n_frames, nc))
        
        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        
        # Step 1: Loop
        for t in range(Tx):
            # Step A: Get the input for time "t"
            x = Lambda(lambda x_input: x_input[:, t, :, :, :])(X)
            
            # Step B: Convolutional part
            ## Convolutional layer (10 filters, 3x3)
            Z1 = Conv2D(10, 3, activation='tanh')(x)       
            ## Maxpooling layer (size = 3, stride = 2)
            max1 = MaxPooling2D(pool_size=3, strides=2, padding='valid')(Z1)
            ## FLATTEN
            F = Flatten()(max1)
            
            # Step C: LSTM part
            ## Reshape x to be (1, n_features)
            x = Reshape((1, F.shape[1]))(F)
            ## Step Perform one step of the LSTM_cell
            a, _, c = LSTM(n_a, return_state = True)(x, initial_state=[a, c])
            
        # Step 2: Apply densors to the hidden state output of LSTM_Cell after iterating through time
        D1 = Dense(30, activation='relu')(a)
        D1 = Dropout(0.25)(D1)
        out = Dense(n_o, activation='softmax')(D1)
            
        # Step 3: Create model instance
        model = Model([X, a0, c0], out)
        
        return model
    
    def _conv_gru_model(self, Tx, n_a, n_w, n_sens, n_o):
        """
        
        TRUE ConvGRU model following the original formulation.
        Uses convolutional operations for all gates.
        
        Arguments:
            Tx (int): Number of timesteps.
            n_a (int): The number of activations used in this model.
            n_w (int): Temporal samples per timestep.
            n_sens (int): Number of sensors (input channels).
            n_o (int): Number of output classes.
        
        Returns:
            model (keras model): True ConvGRU model.
        
        Developer/s:
            Samer A. Mohamed.
            ConvGRU formulation from: Shi et al., "Convolutional LSTM Network"

        """ 

        # Input shape: (batch, timesteps, temporal_samples, sensors)
        # For ConvGRU, we treat this as: (batch, timesteps, height=1, width=n_w, channels=n_sens)
        X = Input(shape=(Tx, n_w, n_sens))
        
        # Add height dimension for Conv2D: (batch, timesteps, 1, n_w, n_sens)
        X_expanded = Reshape((Tx, 1, n_w, n_sens))(X)
        
        # ====== CONVGRU LAYER IMPLEMENTATION ======
        # Take first timestep
        first_timestep = Lambda(lambda x: x[:, 0:1, :, :])(X)

        # 2. Conv2D to get n_a channels
        h = Conv2D(n_a, (1, 1), padding='same', use_bias=False,
                kernel_initializer='zeros')(first_timestep)
        
        # Process sequence through time
        all_hidden = []
        
        for t in range(Tx):
            # Current input frame
            x_t = Lambda(
                lambda x: x[:, t, :, :, :],
                output_shape=(1, n_w, n_sens)  # This is correct for your case
            )(X_expanded)
            
            # ConvGRU equations with Conv2D operations:
            
            # 1. Concatenate input and hidden state along channel dimension
            # x_t: (batch, 1, n_w, n_sens), h: (batch, 1, n_w, n_a)
            # combined: (batch, 1, n_w, n_sens + n_a)
            combined = Concatenate(axis=-1)([x_t, h])
            
            # 2. Reset gate: r_t = σ(Conv(W_r * [h_{t-1}, x_t] + b_r))
            r = Conv2D(filters=n_a, kernel_size=(1, 3), padding='same',
                    activation='sigmoid')(combined)
            
            # 3. Update gate: z_t = σ(Conv(W_z * [h_{t-1}, x_t] + b_z))
            z = Conv2D(filters=n_a, kernel_size=(1, 3), padding='same',
                    activation='sigmoid')(combined)
            
            # 4. Candidate hidden state: ñ_t = tanh(Conv(W_n * [r_t ⊙ h_{t-1}, x_t] + b_n))
            r_h = Multiply()([r, h])
            combined_reset = Concatenate(axis=-1)([x_t, r_h])
            n_t = Conv2D(filters=n_a, kernel_size=(1, 3), padding='same',
                        activation='tanh')(combined_reset)
            
            # 5. New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t
            one_minus_z = Lambda(lambda x: 1 - x, output_shape=lambda input_shape: input_shape)(z)
            h = Add()([
                Multiply()([one_minus_z, h]),
                Multiply()([z, n_t])
            ])
            
            all_hidden.append(h)
        
        # Get final hidden state (last timestep)
        last_hidden = all_hidden[-1]                    # (batch, 1, n_w, n_a)
        
        # Global pooling over spatial dimensions
        pooled = GlobalAveragePooling2D()(last_hidden)  # (batch, n_a)
        
        # Classification
        D1 = Dense(30, activation='relu')(pooled)
        D1 = Dropout(0.25)(D1)
        out = Dense(n_o, activation='softmax')(D1)
        
        model = Model(inputs=X, outputs=out)
        
        return model
    
    def st_gcn(self, Tx, n_w, n_sens, n_o):
        """
        
        Simple ST-GCN with predefined adjacency for 6 IMU sensors.
        
        Arguments:
            Tx (int): Timesteps.
            n_w (int): Samples per timestep.
            n_sens (int): Sensory channels (must be 6).
            n_o (int): Output classes.
        
        Returns:
            model (keras model): ST-GCN model.
        
        Developer/s:
            Samer A. Mohamed.

        """ 

        # Validate input
        if n_sens != 6:
            raise ValueError(f"This implementation expects 6 imu channels (2×3), got {n_sens}")
        
        X = Input(shape=(Tx, n_w, n_sens))
        
        # ====== HARDCODED ADJACENCY FOR 6 IMU SENSORS ======
        # Sensor order: [R_ax, R_az, R_gy, L_ax, L_az, L_gy]
        
        # Create adjacency with rules:
        # 1 = strong connection, 0.5 = weak connection, 0 = no connection
        adjacency = np.array([
            # R_ax, R_az, R_gy, L_ax, L_az, L_gy
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # R_ax
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # R_az
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # R_gy
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # L_ax
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],  # L_az
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # L_gy
        ], dtype=np.float32)
        
        # Convert to tensor
        A = tf.constant(adjacency, dtype=tf.float32)
        
        # Process sequence
        timestep_features = []
        
        for t in range(Tx):
            X_t = Lambda(
                lambda x: x[:, t, :, :], 
                output_shape=(n_w, n_sens)  # From (batch, n_w, 6) after slicing
            )(X)
            
            # Graph convolution
            X_t_T = Permute((2, 1))(X_t)  # (batch, 6, n_w)
            graph_out = Lambda(
                lambda inputs: inputs[0] @ inputs[1],  # Python @ operator
                output_shape=(6, n_w)
            )([A, X_t_T])
            graph_out_back = Permute((2, 1))(graph_out)  # (batch, n_w, 6)
            
            # Temporal processing
            temp_features = Conv1D(32, 3, padding='same', activation='tanh')(graph_out_back)
            pooled = GlobalAveragePooling1D()(temp_features)
            timestep_features.append(pooled)
        
        # Temporal modeling across sequence
        sequence = Concatenate(axis=1)(timestep_features)  # Concatenate all features
        sequence = Reshape((Tx, -1))(sequence)  # Reshape to (batch, Tx, features)
        temporal = Conv1D(64, 9, padding='same', activation='tanh')(sequence)
        temporal = BatchNormalization()(temporal)
        
        # Classification
        pooled_final = GlobalAveragePooling1D()(temporal)
        D1 = Dense(30, activation='relu')(pooled_final)
        D1 = Dropout(0.25)(D1)
        out = Dense(n_o, activation='softmax')(D1)
    
        return Model(inputs=X, outputs=out)
    
    def vanilla_transformer(self, Tx, n_w, n_sens, n_o):
        """

        Vanilla Transformer encoder: standard Transformer for sequence classification.
        
        Arguments:
            Tx (int): Number of timesteps in sequence.
            n_w (int): Number of window samples.
            n_sens (int): Number of sensors/channels.
            n_o (int): Number of output units/classes.
        
        Returns:
            model (keras model): Transformer keras model.
        
        Raises:
            N/A.

        Developer/s:
            Samer A. Mohamed.

        """ 

        # Define the input of your model
        # Shape: (batch, timesteps, samples, sensors)
        X = Input(shape=(Tx, n_w, n_sens))
        
        # Step 1: Reshape to standard Transformer input
        # Combine window_samples and sensors into features dimension
        # From (Tx, n_w, n_sens) to (Tx, n_w * n_sens)
        x = Reshape((Tx, n_w * n_sens))(X)
        
        # Step 2: Project to Transformer dimension (d_model = 64)
        d_model = 64
        x = Dense(d_model)(x)
        
        # Step 3: Add sinusoidal positional encoding (vanilla)
        # Create positional indices
        position = tf.range(Tx, dtype=tf.float32)
        position = tf.reshape(position, [1, -1, 1])  # Shape: (1, Tx, 1)
        
        # Calculate division term for sinusoidal frequencies
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * 
            -(tf.math.log(10000.0) / d_model)
        )
        
        # Initialize positional encoding matrix
        pe = tf.zeros((1, Tx, d_model))
        
        # Fill even indices with sin
        for i in range(Tx):
            for j in range(0, d_model, 2):
                pe = tf.tensor_scatter_nd_update(
                    pe,
                    [[0, i, j]],
                    [tf.sin(position[0, i, 0] * div_term[j // 2])]
                )
        
        # Fill odd indices with cos
        for i in range(Tx):
            for j in range(1, d_model, 2):
                pe = tf.tensor_scatter_nd_update(
                    pe,
                    [[0, i, j]],
                    [tf.cos(position[0, i, 0] * div_term[j // 2])]
                )
        
        # Add positional encoding to input
        x = x + pe
        
        # Step 4: Vanilla Transformer Encoder (2 layers, 4 heads each)
        # Layer 1
        # Multi-head self-attention
        attn_output1 = MultiHeadAttention(num_heads=4, key_dim=d_model)(x, x)
        # Add & LayerNorm
        x = LayerNormalization()(x + attn_output1)
        # Feed-forward network
        ffn_output1 = Dense(d_model * 4, activation='relu')(x)
        ffn_output1 = Dense(d_model)(ffn_output1)
        # Add & LayerNorm
        x = LayerNormalization()(x + ffn_output1)
        
        # Layer 2
        # Multi-head self-attention
        attn_output2 = MultiHeadAttention(num_heads=4, key_dim=d_model)(x, x)
        # Add & LayerNorm
        x = LayerNormalization()(x + attn_output2)
        # Feed-forward network
        ffn_output2 = Dense(d_model * 4, activation='relu')(x)
        ffn_output2 = Dense(d_model)(ffn_output2)
        # Add & LayerNorm
        x = LayerNormalization()(x + ffn_output2)
        
        # Step 5: Global average pooling over timesteps
        pooled = GlobalAveragePooling1D()(x)
        
        # Step 6: Classification head (matches your other models)
        D1 = Dense(30, activation='relu')(pooled)
        D1 = Dropout(0.25)(D1)
        out = Dense(n_o, activation='softmax')(D1)
        
        # Step 7: Create model instance
        model = Model(inputs=X, outputs=out)
        
        return model
    
    def train_model(self):
        """

        Training procedure: trains the model given class parameters.
        
        Args:
            N/A.
        
        Returns:
            N/A.
        
        Raises:
            Error: Insufficient training data.
    
        Developer/s:
            Samer A. Mohamed. 

        """ 
                    
        # Divide CSV files among training, validation and testing portions
        print("LOG1: analyzing dataset....\n")   
        csv_file_names = sorted([f for f in os.listdir(self.ds_path) if f.endswith('.csv')]) # Locate CSV files
        try:
            test_list = [item for item in csv_file_names if item[2:5] == self.name] # Unseen test subject files
            seen_list = [item for item in csv_file_names if item not in test_list] # Seen subject files
            train_list = []; val_list = []
            while len(seen_list):
                seen_list_per_sub = [item for item in seen_list if item[2:5] == seen_list[0][2:5]] 
                train_list.extend(seen_list_per_sub[:int(0.6*len(seen_list_per_sub))]) # ~60% training
                val_list.extend(seen_list_per_sub[int(0.6*len(seen_list_per_sub)):]) # ~40% validation
                seen_list = [item for item in seen_list if item not in seen_list_per_sub]
            if not len(train_list) or not len(val_list):
                raise ValueError("The CSV files within the dataset are not sufficient for training," \
                                 " collect more data from seen subjects.")
        except ValueError as e:
            print(f"ERROR: {str(e)}")
            sys.exit()
            
        # Cut the CSV files into separate continuous gait cycles
        train_cut = self._DSpcut(train_list)
        val_cut = self._DSpcut(val_list)
        test_cut = self._DSpcut(test_list)
        print("LOG2: dataset segmented successfully!\n")
        
        # Preprocess the dataset to extract meaningful features/labels
        print("LOG3: extracting features and labels....\n")
        X_train, y_train = self._DSpprocess(train_cut)
        X_val, y_val = self._DSpprocess(val_cut)   
        X_test, y_test = self._DSpprocess(test_cut)   
        print("LOG4: dataset processed successfully!\n")
        
        # Create a keras model handle
        print("LOG5: creating the model handle...\n")
        out_sz = len(np.unique(y_train))
        if self.trainer == "lstm":
            self.n_act = 64 # Number of activations
            self.tmodel = self._lstm_model(Tx = self.s, n_a = self.n_act, n_values = self.w*len(self.modalities), n_o = out_sz)
            self.tmodel.compile(optimizer=Adadelta(learning_rate=0.05),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            self.tmodel.summary()
        elif self.trainer == "cnn-lstm":
            self.n_act = 60 # Number of activations
            self.tmodel = self._cnn_lstm_model(Tx = self.s, n_a = self.n_act, n_sens = len(self.modalities), 
                                   n_frames = self.w, nc = 1, n_o = out_sz)
            self.tmodel.compile(optimizer=Adadelta(learning_rate=0.05),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            self.tmodel.summary()
        elif self.trainer == "convGRU":
            # Build the ConvGRU model
            self.n_act = 32 # Number of activations
            self.tmodel = self._conv_gru_model(
                Tx=self.s,
                n_a = self.n_act,
                n_w=self.w,
                n_sens=len(self.modalities),
                n_o=out_sz 
            )
            self.tmodel.compile(
                optimizer=Adadelta(learning_rate=0.05),  # Same as your CNN-LSTM
                loss='categorical_crossentropy',  # Use sparse if labels are integers
                metrics=['accuracy']
            )
            self.tmodel.summary()
        elif self.trainer == "gnn":
            # Assuming you have 6 IMU sensors/modalities
            if len(self.modalities) != 6:
                print(f"Warning: ST-GCN expects 6 IMU channels, but got {len(self.modalities)}")
                print("Consider selecting exactly 6 features from your data")
            self.tmodel = self.st_gcn(
                Tx=self.s,           # Number of windows within a sequence
                n_w=self.w,          # Window size
                n_sens=len(self.modalities),  # Sensory channels (should be 6)
                n_o=out_sz           # Output classes
            )
            self.tmodel.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.tmodel.summary()
        elif self.trainer == "transformer":
            # Assuming you have 6 IMU sensors/modalities
            if len(self.modalities) != 6:
                print(f"Warning: Transformer expects 6 IMU channels, but got {len(self.modalities)}")
                print("Consider selecting exactly 6 features from your data")
            self.tmodel = self.vanilla_transformer(
                Tx=self.s,           # Number of windows within a sequence
                n_w=self.w,          # Window size
                n_sens=len(self.modalities),  # Sensory channels (should be 6)
                n_o=out_sz           # Output classes
            )
            self.tmodel.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.tmodel.summary()

        print("LOG6: Model handle created successfully!\n")
        
        # Convert labels to one-hot representation
        y_train = self._convert_to_one_hot(y_train)
        y_val = self._convert_to_one_hot(y_val)
        y_test = self._convert_to_one_hot(y_test)
        
        # Training process   
        if self.trainer == "lstm" or self.trainer == "cnn-lstm":
            # Fit the model
            a0 = np.zeros((X_train.shape[0], self.n_act))
            c0 = np.zeros((X_train.shape[0], self.n_act))
            a0_val = np.zeros((X_val.shape[0], self.n_act))
            c0_val = np.zeros((X_val.shape[0], self.n_act))
            history = self.tmodel.fit([X_train, a0, c0], y_train, epochs=10, validation_data=([X_val, a0_val, c0_val], y_val))
            
            # Comment on unseen testing accuracy
            a0_test = np.zeros((X_test.shape[0], self.n_act))
            c0_test = np.zeros((X_test.shape[0], self.n_act))
            test_acc = 1 - (np.count_nonzero(self._convert_from_one_hot(self.tmodel.predict_on_batch([X_test, a0_test, c0_test])) \
                                             - self._convert_from_one_hot(y_test)) / y_test.shape[0]) #--> Compute test accuracy on unseen data
            print("test_accuracy: " + str(test_acc))
            
            # Save the data to a JSON file
            with open(self.save_dir + '/' + self.trainer + '_model_W_' + self.name + '_' + str(self.target_fs) + 'Hz.json', 'w') as file:
                model_meta = {"win_sz": self.w,
                               "n_act": self.n_act,
                               "seq_sz": self.s,
                               "modalities": self.modalities,
                               "phases": self.phases}
                json.dump(model_meta, file)
        elif self.trainer == "convGRU" or self.trainer == "gnn" or self.trainer == "transformer":
            # Fit the model
            history = self.tmodel.fit(
                X_train, y_train, 
                epochs=10, 
                validation_data=(X_val, y_val)
            )

            test_acc = 1 - (np.count_nonzero(self._convert_from_one_hot(self.tmodel.predict_on_batch(X_test)) \
                                             - self._convert_from_one_hot(y_test)) / y_test.shape[0]) #--> Compute test accuracy on unseen data
            print("test_accuracy: " + str(test_acc))
            
            # Save the data to a JSON file
            with open(self.save_dir + '/' + self.trainer + '_model_W_' + self.name + '_' + str(self.target_fs) + 'Hz.json', 'w') as file:
                model_meta = {"win_sz": self.w,
                               "seq_sz": self.s,
                               "modalities": self.modalities,
                               "phases": self.phases}
                if self.trainer == "convGRU":
                    model_meta["n_act"] = self.n_act
                json.dump(model_meta, file)
        
        # Plot learning curves
        df_loss_acc = pd.DataFrame(history.history)
        df_loss= df_loss_acc[['loss','val_loss']]
        df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
        df_acc= df_loss_acc[['accuracy','val_accuracy']]
        df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
        df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
        df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
        
        # Save the model
        self.tmodel.save(self.save_dir + '/' + self.trainer + '_model_W_' + self.name + '_' + str(self.target_fs) + 'Hz.keras')
        
    def train_model_kfolds(self):
        """

        K-fold training: trains the model given class parameters using multiple folds.
        
        Args:
            N/A.
        
        Returns:
            N/A.
        
        Raises:
            Error: Empty dataset.
    
        Developer/s:
            Samer A. Mohamed.  

        """ 
        # Investigate unique subjects in the dataset
        names = list(set([f[2:5] for f in os.listdir(self.ds_path) if f.endswith('.csv')]))
        if names:
            # Loop over different folds by selecting all possible unseen subjects
            for i in range(len(names)):
                self.set_unseen_sub(int(names[i])) # Set a new unseen code name
                self.train_model()
        else:
            # Display an error: empty dataset
            print("ERROR: the provided dataset directory is empty, please choose a valid directory.")
            sys.exit()