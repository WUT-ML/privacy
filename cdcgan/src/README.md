# Generative adversarial network


### gan_main.py

This function implements Conditional DCGAN using PyTorch. Works for fingerprint dataset only.

The parameters of the function are:

- `--mode`: `train` or `sample`
- `--num_epochs`: max number of learning epochs, by default = 100.
- `--image_size`: size of input image, default 512x512 (size of output image may be different)
- `--model_path`: path where trained models are saved, by default '../models'
- `--sample_path`: path where samples are saved, by default '../samples'
- `--image_path`: path where training images are stored, by default '../data/train'
- `--number_of_samples`: number of samples generated for each class
- `--save_only_last_model`: if true then intermediate models are not saved, by default false
