# Using the InnerEye code as a git submodule of your project

You can use InnerEye as a submodule in your own project. 
If you go down that route, here's the list of files you will need in your project (that's the same as those
given in [this document](building_models.md))
* `environment.yml`: Conda environment with python, pip, pytorch
* `settings.yml`: A file similar to `InnerEye\settings.yml` containing all your Azure settings
* A folder like `ML` that contains your additional code, and model configurations.
* A file like `myrunner.py` that invokes the InnerEye training runner, but that points the code to your environment 
and Azure settings; see the [Building models](building_models.md) instructions for details. Please see below for how
`myrunner.py` should look like.

You then need to add the InnerEye code as a git submodule, in folder `innereye-deeplearning`:
```shell script
git submodule add https://github.com/microsoft/InnerEye-DeepLearning innereye-deeplearning
```
Then configure your Python IDE to consume *both* your repository root *and* the `innereye-deeplearning` subfolder as inputs.
In Pycharm, you would do that by going to Settings/Project Structure. Mark your repository root as "Source", and 
`innereye-deeplearning` as well.

Example commandline runner that uses the InnerEye runner (called `myrunner.py` above):
```python
import sys
from pathlib import Path


# This file here mimics how the InnerEye code would be used as a git submodule. 

# Ensure that this path correctly points to the root folder of your repository.
repository_root = Path(__file__).absolute()


def add_package_to_sys_path_if_needed() -> None:
    """
    Checks if the Python paths in sys.path already contain the /innereye-deeplearning folder. If not, add it.
    """
    is_package_in_path = False
    innereye_submodule_folder = repository_root / "innereye-deeplearning"
    for path_str in sys.path:
        path = Path(path_str)
        if path == innereye_submodule_folder:
            is_package_in_path = True
            break
    if not is_package_in_path:
        print(f"Adding {innereye_submodule_folder} to sys.path")
        sys.path.append(str(innereye_submodule_folder))


def main() -> None:
    try:
        from InnerEye import ML  # noqa: 411
    except:
        add_package_to_sys_path_if_needed()

    from InnerEye.ML import runner
    print(f"Repository root: {repository_root}")
    # Check here that yaml_config_file correctly points to your settings file
    runner.run(project_root=repository_root,
               yaml_config_file=Path("settings.yml"),
               post_cross_validation_hook=None)


if __name__ == '__main__':
    main()

```

## Adding new models

1. Set up a directory outside of InnerEye to holds your configs. In your repository root, you could have a folder
`InnerEyeLocal`, parallel to the InnerEye submodule, alongside `settings.yml` and `myrunner.py`.

The example below creates a new flavour of the Glaucoma model in `InnerEye/ML/configs/classification/GlaucomaPublic`. 
All that needs to be done is change the dataset. We will do this by subclassing GlaucomaPublic in a new config 
stored in `InnerEyeLocal/configs`
1. Create folder `InnerEyeLocal/configs`
1. Create a config file called GlaucomaPublicExt.py there which extends the GlaucomaPublic class that looks like
```python
from InnerEye.ML.configs.classification.GlaucomaPublic import GlaucomaPublic


class GlaucomaPublicExt(GlaucomaPublic):
    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id="name_of_your_dataset_on_azure"
``` 
1. In `settings.yml`, set `model_configs_namespace` to `InnerEyeLocal.configs` so this config  
is found by the runner. Set `extra_code_directory` to `InnerEyeLocal`.

#### Start Training
Run the following to start a job on AzureML: 
```
python myrunner.py --azureml=True --model=GlaucomaPublicExt
```
See [Model Training](building_models.md) for details on training outputs, resuming training, testing models and model ensembles.
