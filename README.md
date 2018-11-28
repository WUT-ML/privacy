## Research project: Utility-preserving data privatization

<a href="https://circleci.com/gh/WUT-ML/privacy"><img src="https://circleci.com/gh/WUT-ML/privacy.svg?style=svg&circle-token=50a67db445aab324ef969f27d5fd365ff9b20b7d" align="right"></a>

<img src="http://forthebadge.com/images/badges/made-with-python.svg" />

> ###### [Overview](#overview) | [Requirements](#requirements) | [Steps](#steps) 


## Overview

This repository contains code for reproducing the experiments with a Generative Adversarial Privatizer
, based on the [Siamese Generative Adversarial Privatizer for Biometric Data](https://arxiv.org/pdf/1804.08757.pdf).
Note that the NIST Special Database 4 (FIGS), of the fingerprints,
has been [withdrawn from public use](https://www.nist.gov/srd/nist-special-database-4),
thus it is not used in this repository.

## Requirements

###### • Python 3.5
###### • PyTorch 0.3 & torchvision


## Steps

###### 0. Setup

```bash
pip install -r requirements.txt
```

###### 1. Dataset preparation

Two datasets, FERG and CelebA can be downloaded and extracted by only setting `--dataset` argument to either "FERG" or "CelebA".
For other datasets new dataloaders have to be written.

###### 2. Siamese GAN training

Run in background/separate session:

```bash
tensorboard --logdir runs
```

Train model on FERG dataset with TensorBoard visualizations:

```bash
python src/generative_siamese/generator_plus_siamese_main.py --tensorboard  
```
