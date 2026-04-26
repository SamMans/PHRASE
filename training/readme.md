This folder contains the training code for PHRASE, as well as the dataset used to train the model

Kindly note that the dataset does not represent the real gait phase events, but rather the transitions associated with some critical points such as peaks and zero crossings

If you want to create your own training dataset, you have to read the paper to understand the instructions for heuristic transitions and labeling, although this is unnecessary as the method is supposed to generalize to external datasets anyway

The trained model will be generated as a .json file that can then be used for benchmarking (check the benchmarking folder in the main folder)

The training strategy trains the model several times by training the model on data from seen subjects, validating on held-out data from the same seen subjects, then testing on held-out data from one unseen subject (i.e, if we have ten subjects, the model will be trained 10 times, generating 10 pretrained models)

To understand why the training strategy was designed this way, please refer to the paper

To start the training procedure:

1- open MATLAB

2- navigate to the directory which contains this readme file on MATLAB

3- run the code trainAll.m by typing trainAll.m in the command window, or clicking on trainAll.m file in current folder, which will open the code, and then click the 'run' button on the toolbar

4- wait till training is finalized, and the pretrained models will automatically be saved to the checkpoints folder in the main project folder
