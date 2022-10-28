# Speaker Recognition using x vectors

This is my speaker recognintion implementation based on the x-vector system described in "X-Vectors: Robust DNN Embeddings for Speaker Recognition" by Snyder et al. I developed this programm as part of my bachelor thesis. if you are interested in the theory of how this works you can read the paper this implementation is based on here: https://ieeexplore.ieee.org/document/8461375

## File Overview

 - `main.py` the main file of this project which also contains the definition of the x-vector model. run this to execute the programm.
 - `config.py` the config file for this project. If you want to change certain parameters I recommend doing so when config is called inside `main.py`. there you can change the parameters you want without havin to edit this file.
 - `dataset.py` this file handels the dataset. it controls the split into train, validation and test as well as the data augmentation.
 - `tdnn_layer.py` this holds tha class which defines the time context layers which are used in the x-vector model.
 - `plda_classifier` the plda classifier of the x-vector system. this mostly just contains some utility functions to make calling SpeechBrains plda functions easier. it also includes a sae fuction to save trained plda classifiers.
 - `plda_score_stat` this files class handles the score calculation and evaluation of the results. it also generates plots for the tensorboard logger.

## How to use

First you need to have the 3 datasets downloaded (plus the extra VoxCeleb test file):
 - VoxCeleb (main dataset): https://github.com/clovaai/voxceleb_trainer (follow the Data preparation steps in the gits README to download the dataset)
 - MUSAN (noise augmentation): https://www.openslr.org/17/
 - RIR (noise augmentation): https://www.openslr.org/28/
 - `veri_test2.txt`: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt>

All 3 datasets schould be in the same data folder. afterward you have to give the path to the dataforlder to config when it is called inside `main.py`. here you can also change other parametres for running the program like the batch size, the learning rate etc. The programm assumes the data folder structure to look like this:

```
.
└── data/
    ├── VoxCeleb/
    |   ├── vox1_dev_wav/
    |   ├── vox1_test_wav/
    |   └── veri_test2.txt
    ├── musan/
    |   ├── music/
    |   ├── noise/
    |   └── speech/
    └── rir_noises
        └── simulated_rirs/
```

The system works in 4 main steps:
 1. Training the neural network
 2. generating the x-vectors
 3. trainng the plda
 4. testing the plda

if you just want to run the whole program just execute `main.py`.

in order to execute only certain parts of the programm you can switch the four config variables: `train_x_vector_model`, `extract_x_vectors`, `train_plda` and `test_plda` between `True` and `False`. for example if you only want to train the network for now and extract the x-vectors later set `train_x_vector_model` to `True` and all others to `False`.

when executing later staps of the programm or when you want to continue training a model that isnt finisched jet, you have to give a checkpoint path to config when calling it in main. these checkpoints are automatically generated when training the model and are saved in a lokally generated folder called `lightning_logs/`. the output of the other steps is also saved in locally generated folders. the extracted x-vectors will be save in '.csv' files in a 'x_vectors/' folder and the trained plda classifier and plda score stat object are saved as '.pickle' files in a 'plda/' folder.

if you want to generate multiple different sets of x-vectors and train different plda classifiers, you have to manually adjust the names of the created files and the loading folderpaths inside 'main.py'.

## How it works
### (Detailed step-by-step runthrough of the whole programm)

TODO
