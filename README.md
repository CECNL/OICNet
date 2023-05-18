# OICNet
This repository is the official implementation of "OICNet: Online Independent Component Analysis Network for EEG Source Separation".
## Requirements
* To install requirements:
```
conda env create -f OICNet.yaml
conda activate OICNet
```
## Usage
* Here is an example command
```
python train_bcic.py --subj "S02" --alpha 0.994 --epoch 5 --lr 0.001 --save 1 --save_path ./result 
```
### optional arguments:
* `--lr`                    Learning rate
* `-a`, `--alpha`           Coefficient of contrast function
* `-e`, `--epoch`           Number of epoch for fine-tuning
* `--ff`                    Forgetting factor for RLS Whitening
* `--bs`                    Block size for RLS whitening
* `-s`,  `--save`           Save OICNet
* `-p`, `--save_path`       Specify the path to save model
* `--subj`                  Specify the subject of training data
* `--session`               Specify the session of training data
* `--init`                  Path of initial weights file for OICNet
* `--filename`              Postfix for file name
* `--verbose`               Show system message
* `--record`                Record the model after each fine-tuning
## Repository Structure and Descriptions
* The directory structure of this repository is listed as below:
```
.
├── isctest
│   ├── computeSimilarities.py
│   ├── isctest.py
│   └── uppertriangindices.py
├── model.py
├── function.py
├── train_bcic.py
├── train_yoto.py
└── util.py
```
### Descriptions
#### Root
* `model.py` : Implementation of OICNet class.
* `function.py` : Implementations of numerical operation.
* `util.py` : Tools for analysis such as plotting, performance evaluation and component match, etc.
* `train_bcic.py` : Sample Code to run OICNet on BCI Challenge @ Ner 2015 dataset.
#### isctest
* Python version of implementation of IC test algorithm [1]
## Dataset
* The full dataset of BCI Challenge @ Ner 2015 dataset is available on [Kaggle](https://www.kaggle.com/c/inria-bci-challenge).
## References
[1] Hyvärinen, Aapo. "Testing the ICA mixing matrix based on inter-subject or inter-session consistency." NeuroImage 58.1 (2011): 122-136.
