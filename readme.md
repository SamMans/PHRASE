This repo provides the source code for PHRASE (Probabilistic Heuristic Recognition Algorithm for Sequential Events)

The training code for PHRASE is written using MATLAB scripting language in the 'training' folder. Go to that folder to download the training code and run it on MATLAB directly. The code was developed using MATLAB R2024B on Windows 10.

The 'benchmarking' folder contains Python code for PHRASE inference on real-world datasets plus benchmarking codes for other similar methods that are compared against PHRASE (Transformers, GNNs, RNNs, etc.)

The Python code was developed in a conda virtual environment on Ubunto 22.04 LTS. All the dependencies can be installed by downloading the project package that contains this readme file 'PHRASE', navigating to the PHRASE directory, and running this terminal line: conda env create -f tf_env.yml

Before running any of the Python files, you have to activate the conda environment 'tf' in the terminal: conda activate tf

The 'resources' folder contains the dataset-specific csv files that were used in this project

The 'checkpoints' folder contains the benchmark pre-trained models, which can be recreated using the training code in the 'benchmarking' folder
