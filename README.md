## Research project: Utility-preserving data privatization

<a href="https://circleci.com/gh/WUT-ML/privacy"><img src="https://circleci.com/gh/WUT-ML/privacy.svg?style=svg" align="right"></a>

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
