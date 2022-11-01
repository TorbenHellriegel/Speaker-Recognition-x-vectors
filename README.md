# Speaker Recognition using X-Vectors

This is my speaker recognition implementation based on the x-vector system described in "X-Vectors: Robust DNN Embeddings for Speaker Recognition" by Snyder et al. I developed this program as part of my bachelor thesis. If you are interested in the theory of how this works, you can read the paper this implementation is based on here: <https://ieeexplore.ieee.org/document/8461375>

## File Overview

 - `main.py` The main file of this project which also contains the definition of the x-vector model. Run this to execute the program.
 - `config.py` The config file for this project. If you want to change certain parameters, I recommend doing so when config is called inside `main.py`. There you can change the parameters you want without having to edit this file.
 - `dataset.py` This file handles the dataset. It controls the split into training, validation and test as well as the data augmentation.
 - `tdnn_layer.py` This holds the class which defines the time context layers which are used in the x-vector model.
 - `plda_classifier` The PLDA classifier of the x-vector system. This mostly just contains some utility functions to make calling SpeechBrains PLDA functions easier. It also includes a save function to save trained PLDA classifiers.
 - `plda_score_stat` This files class handles the score calculation and evaluation of the results. it also generates plots for the TensorBoard logger.

## How to use

First you need to have the three datasets downloaded (plus the extra VoxCeleb test file):
 - VoxCeleb (main dataset): <https://github.com/clovaai/voxceleb_trainer> (Follow the Data preparation steps in the gits README to download the dataset.)
 - MUSAN (noise augmentation): <https://www.openslr.org/17/>
 - RIR (noise augmentation): <https://www.openslr.org/28/>
 - `veri_test2.txt`: <https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt>

All three datasets should be in the same data folder. Afterward, you have to give the path to the data folder to config when it is called inside `main.py`. Here you can also change other parametres for running the program like the batch size, the learning rate, etc. The program assumes the data folder structure to look like this:

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

The system works in four main steps:
 1. Training the neural network
 2. Generating the x-vectors
 3. Training the PLDA
 4. Testing the PLDA

If you just want to run the whole program just execute `main.py`.

In order to execute only certain parts of the program you can switch the four config variables: `train_x_vector_model`, `extract_x_vectors`, `train_plda` and `test_plda` between `True` and `False`. For example if you only want to train the network for now and extract the x-vectors later set `train_x_vector_model` to `True` and all others to `False`.

When executing later steps of the program or when you want to continue training a model that isn't finished jet, you have to give a checkpoint path to config when calling it in main. These checkpoints are automatically generated when training the model and are saved in a locally generated folder called `lightning_logs/`. The output of the other steps is also saved in locally generated folders. The extracted x-vectors will be saved in `.csv` files in a `x_vectors/` folder and the trained PLDA classifier and PLDA score stat object are saved as `.pickle` files in a `plda/` folder.

If you want to generate multiple different sets of x-vectors and train different PLDA classifiers, you have to manually adjust the names of the created files and the loading folder paths inside `main.py`.

## Credit

The SpeechBrain toolkit was used in this implementation for the PLDA classifier as well as calculating the EER and DCF.
```
M. Ravanelli et al. ”SpeechBrain: A general-purpose speech toolkit”.
https://arxiv.org/abs/2106.04624, 2021. 
```
