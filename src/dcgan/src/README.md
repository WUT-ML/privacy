# Generative adversarial network


### gan_main.py

This function implements DCGAN using PyTorch

The parameters of the function are:

- `--mode`: `train` or `sample`
- `--num_epochs`: max number of learning epochs, by default = 100.
- `--image_size`: size of input image, default 512x512 (size of output image maybe different)
- `--model_path`: path where trained models are saved, by deafault '../models'
- `--sample_path`: path where samples are saved, by default '../samples'
- `--image_path`: path where training images are stored, by default '../data/train'
- `--save_only_last_model`: if true then intermediate models are not saved, by default false
