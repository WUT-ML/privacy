## Research project: Utility-preserving data privatization

<a href="https://circleci.com/gh/WUT-ML/privacy"><img src="https://circleci.com/gh/WUT-ML/privacy.svg?style=svg&circle-token=50a67db445aab324ef969f27d5fd365ff9b20b7d" align="right"></a>

<img src="http://forthebadge.com/images/badges/made-with-python.svg" />

> ###### [Overview](#overview) | [Requirements](#requirements) | [Steps](#steps) 


## Overview

TBD


## Requirements

###### • Python 3.5
###### • PyTorch 0.3 & torchvision
###### • luigi


## Steps

###### 0. Setup

```bash
pip install -r requirements.txt
```

###### 1. Dataset preparation

```bash
python ./run.py Dataset --local-scheduler
```

Download & extract the [NIST FIGS database](https://www.nist.gov/srd/nist-special-database-4) to `data/` folder.

Images are contained in `data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_*` folders, 250 pairs in each.

File naming is `{instance}{ID}_{finger}.png`, where:

- `{instance}` - either `f` or `s` for first and second instance,
- `{ID}` - person identifier,
- `{finger}` - finger identifier `[01-10]`.

###### 2. Siamese GAN training

Run in background/separate session:

```bash
tensorboard --logdir runs
```

Train model with TensorBoard visualizations:

```bash
python src/generative_siamese/src/generator_plus_siamese_main.py --image_path=data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/ --model_path=results/models --tensorboard  
```

This step will be superseded by a Luigi task in the future.
