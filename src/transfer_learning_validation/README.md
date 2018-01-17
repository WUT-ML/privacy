# Transfer learning validation


### transfer_learning_main.py

The purpose of this module is to perform transfer learning using pretraind ResNet model.

We start with ResNet without last fc layer, then train model (unfreeze all weights) with new data.
The train and validation images should be (by default) in ./data/train and ./data/val divided into
subdirectories: one subdirectory for one class.
At the end of each training epoch model is validated in order to improve classification accuracy.

The parameters of the function are:

- `--num_epochs`: max number of learning epochs, by default = 100.
- `--image_path`: path where training images are stored, by default '../data/train'
