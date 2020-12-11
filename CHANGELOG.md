# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is done, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.
 
## Upcoming

### Added
- New extensions of SegmentationModelBases `HeadAndNeckBase` and `ProstateBase`. Use these classes to build your own Head&Neck or Prostate models, by just providing a list of foreground classes.

### Changed
- The arguments of the `score.py` script changed: `data_root` -> `data_folder`, it no longer assumes a fixed
`data` subfolder. `project_root` -> `model_root`, `test_image_channels` -> `image_files`.
- By default, the visualization of patch sampling for segmentation models will run on only 1 image (down from 5).
This is because patch sampling is expensive to compute, taking 1min per large CT scan.
- Renamed `HeadAndNeckBase` to `HeadAndNeckPaper`, and `ProstateBase` to `ProstatePaper`.

### Fixed
- When registering a model, it now has a consistent folder structured, described [here](docs/deploy_on_aml.md). This
folder structure is present irrespective of using InnerEye as a submodule or not. In particular, exactly 1 Conda
environment will be contained in the model.

### Removed
- Removed blobxfer completely. When downloading a dataset from Azure, we now use AzureML dataset downloading tools.
Please remove the following fields from your settings.yml file: 'datasets_storage_account' and 'datasets_container'. 
- Removed `ProstatePaperBase`.

### Deprecated



## 0.1 (2020-11-13)
- This is the baseline release.
