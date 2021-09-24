# Harnessing the power of GANs: learning segmentations with fewer annotations
#### Margherita Rosnati's internship project, supervised by Daniel Coelho de Castro & the InnerEye team

## Outlook:
This project has three main components:

- a StyleGAN2 transfer learning component
- a pixel-wise classification component
- a UNet segmentation component

A file outline can be found at the bottom of this page.

Note: all scripts are written to be run from the root directory (parent of the folder `InnerEye`).
## Prep step: Prepare data
As a first step, download the prostate dataset and save its directory as `DATASETPATH` in `locations.py`

Genereate the 2D slices: run
```
python InnerEye-Generative/scripts/save2DprotateScans.py
```

Note: `save2DprotateScans`:
- resamples the scans to a voxel size of [4, 4, 4]
- Crops the central 128x128 space
- clips the intensities to [-100, 155], that is standard abdominal intensities
- shifts the resulting intensities in the range [-1, 1] for GAN compatibility
- saves every second slice for 24 central slices in `DATASETPATH / 2D / scans` and `DATASETPATH / 2D / labels` together with a csv file

Note that the `dataset.csv` file's index was subsequently modified so that the first k instances were varied and interesting first k scans for the segmentation model using `InnerEye-Generative/notebooks/legacy____select_k_shots.ipynb`.

Separate patients into a training, validation and test sets: run 
```
python InnerEye-Generative/scripts/prostate2DDataSep.py
```


## Step 0: GAN metrics
There are two example files to calculate GAN metrics:
- InnerEyePrivate/ML/configs/notebooks/metrics_VGG_pretrained.ipynb
- InnerEyePrivate/ML/configs/scripts/metrics_VGG_rand.py

The first is better commented, runs locally and uses a pretrained VGG net to embed scans.
The second is set up to run with hi-ml. The hard-coded commands run the script on InnerEye's cluster - they can be overwritten. The second script uses a randomly initialised VGG net and was used for the end-of-internship presentation.

## Step 1: Run GANs on 2D slices
All of the GANs rely on the dataset `InnerEye-Generative/loaders/Prostate2DSimpleDataset` and are based on pytorch lightning modules
### Step 1.0: DCGAN
```
python InnerEye-Generative/mains/DCGAN_main.py --gpu 0 --azureml
```

Note that this model never performed as well as the StyleGAN2. One of the characteristics we noticed is that the model generated very pixelised scans. We make the hypothesis that this is due to the model being too shallow.

### Step 1.1: StyleGAN2
In order to run StyleGAN2, you will need to clone the [official repo](https://github.com/NVlabs/stylegan2-ada-pytorch) in `stylegan2-ada-pytorch`.
```
git submodule add https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

```
python InnerEye-Generative/mains/StyleGAN2_main.py --gpus 2 --azureml --mean_last
```

## Step 2: Synthetic dataset generation
First save your prefered weights of the StyleGAN model you trained into a directory and refer to it in `locations.py` at `TRAINEDSTYLEGAN2WEIGHTS`.
Indicate in `locations.py` at `GENSCANSPATH` where you would like the scans and latent representations to be saved.

Then run 
```
python InnerEye-Generative/scripts/save_synth_scans_to_segment.py --gpu 0
```

The script will save the latent representations as pickle files in `GENSCANSPATH/{train or val}` and 2D slices as nifti files in `GENSCANSPATH/{train or val}/scans`

- TODO: the script only runs locally atm, needs to be adapted to run on AML
- TODO: I could not check if it's fully functional because my VM no longer has a GPU :( 

The synthetic scans can then be segmented manually using Microsoft Radiomics Dev or your prefered segmentation tool. Note that their format is MRD-friendly.
Once segmented, the segmentations should be saved in `GENSCANSPATH/{train or val}/labels` in the format `{patient number}_{label}.nii.gz`. We used labels `[femur_l, femur_r, bladder, prostate]`.

You can generate plots of the scans and segmentations using 
```
python InnerEye-Generative/scripts/visualisation_gen_scans_n_labels.py
```

## Step 3: MLP: GAN latent rep -> pixel prediction
```
python InnerEye-Generative/mains/seg_StyleGAN2_main.py --gpu 1 --azureml --add_scan_to_latent --n_scans 16
```
- `n_scans` says how many scans are used to train the model. Reminder that each scan corresponds to 128x128 independently handled pixel latent representations.
- `--add_scan_to_latent` is a flag that when activated asks the dataloader to add the generated scan pixel as one of the latent representations of the pixel.

We trained 2x8 pixel-wise segmentation models, using {3, 4, 6, 8, 10, 12, 15, 16} labelled scans [8 cases] and either adding the generated image as a last latent channel or not [2 cases].

Once the model has learned to segment GAN generated images, we create datasets of synthetic images and their corresponding segmentation to train the UNet. 
A script producing these datasets on AML can be run with
```
python InnerEye-Generative/scripts/generate_seg_training_dataset.py --segStyleGAN_folder n_10_scan --azureml
``` 

## Step 4: UNet: synthetic scan -> label
```
python InnerEye-Generative/mains/synth_UNet2D_main.py --gpu 0 --azureml
```

## Step 5: UNet: real scan -> label
```
python InnerEye-Generative/mains/UNet2D_main.py --k_shots 16 --gpu 0 --azureml
```


## File structure
```
generative_models
 ├── assets                                     <- where models and datasets can be saved 
 ├── helpers
 │   └── loggers.py                             <- modified tensorboard and aml loggers to log images throughout the project
 │
 ├── loaders
 │   ├── gen_imgs_and_seg_loader.py             <- Step 4 loader
 │   ├── gen_imgs_loader.py                     <- Step 3 loader
 │   ├── prostate_loader.py                     <- Steps 0, 1 and 5 loader
 │   ├── prostate_loader_3D.py                  <- WIP on 3D part of the project
 │   └── transformation_utils.py                <- Transformations used in step prep
 │
 ├── mains
 │   ├── DCGAN_main.py                          <- Step 1.0 main file
 │   ├── StyleGAN2_3D_main.py                   <- WIP on step 1.1 in 3D main file
 │   ├── StyleGAN2_main.py                      <- Step 1.1 main file
 │   ├── UNet2D_main.py                         <- Step 5 main file
 │   ├── seg_StyleGAN2_main.py                  <- Step 3 main file
 │   └── synth_UNet2D_main.py                   <- Step 4 main file
 │
 ├── metrics
  │   └── get_VGG_model.py                       <- GAN metrics were calculated from a VGG embedding, instantiated here
 │
 ├── models
 │   ├── DCGAN.py                               <- Step 1.0 model
 │   ├── StyleGAN2_ADA_frozen.py                <- Step 1.1 model
 │   ├── StyleGAN2_segment_addon.py             <- Step 3 model
 │   ├── UNet2D_seg_baseline.py                 <- Step 4 and 5 models
 │   └── gan_loss.py                            <- GAN losses used in steps 1.0 and 1.1
 │
 ├── notebooks
 │   ├── metrics_VGG_pretrained.ipynb           <- Step 0 calculation notebook 
 │   ├── legacy____select_k_shots.ipynb         <- Notebook to pick first k patients for step 5. NOTE: has not been adapted to this env.
 │   └── seg_models_metrics.ipynb               <- Notebook to evaluate step 5 models and generate corresponding graph. NOTE: has not been adapted to this env.
 │
 ├── scripts
 │   ├── metrics_VGG_rand.py                    <- Step 0 calculation script
 │   ├── prostate2DDataSep.py                   <- Separating training and validation data for 2D slices (steps 1 and 5)
 │   ├── save2DprotateScans.py                  <- Generating 2D dataset from 3D slices
 │   ├── save_synth_scans_to_segment.py         <- Step 2: generating daset for step 4 (excludes manual segmentations script, which is manual and not python based)
 │   └── visualisation_gen_scans_n_labels.py    <- Visualises the output of manual segmentations produced in step 2.
 │
 ├── .amlignore                                 <- files and folders that AzureML should ignore                              
 ├── README.md                                  <- this file
 ├── environment.yml                            <- required packages
 └── locations.py                               <- locations of files generated. NOTE: to be manually updated as you go along the project
 ```
