# Setting up your environment

We recommend using [WSL2](https://docs.microsoft.com/en-us/windows/wsl/about) for development or Linux with PyCharm or VSCode. Linux is better supported by Pytorch.

## WSL2 setup
- [WSL2 development](/docs/WSL.md)

## PyCharm
- Our team uses [PyCharm](https://www.jetbrains.com/pycharm/) for development.
- Add the contents listed just below in file `InnerEye-DeepLearning.iml` in the `.idea` folder in the repo.
 This will configure the interpreter and modules for PyCharm.
- Change the jdkName to your WSL interpreter. At the moment there is no way to change the name of the interpreter for WSL.
```xml
<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder url="file://$MODULE_DIR$" isTestSource="false" />
    </content>
    <orderEntry type="jdk" jdkName="3.7 @ Ubuntu" jdkType="Python SDK" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
  <component name="PackageRequirementsSettings">
    <option name="requirementsPath" value="" />
  </component>
  <component name="PyDocumentationSettings">
    <option name="format" value="PLAIN" />
    <option name="myDocStringFormat" value="Plain" />
  </component>
  <component name="TestRunnerService">
    <option name="PROJECT_TEST_RUNNER" value="pytest" />
  </component>
</module>
```
- To make sure operations such as git clean do not remove this file, you can add our `deepclean` alias to your local 
git config by applying our git configurations: `git config --local include.path ../.gitconfig`

## How to manually set up flake8 as a PyCharm external tool

Go to File / Settings / Tools / External Tools / Add.

    * Name: Flake8
    * Program: $PyInterpreterDirectory$/python
    * Arguments: -m flake8 $ProjectFileDir$
    * Working directory: $ProjectFileDir$
    * Advanced Options / Output Filters: $FILE_PATH$\:$LINE$\:$COLUMN$\:.*

Run Flake8 by right-clicking on a source file, External Tools / Flake8

## Mypy

* From the command line: `mypy --config-file=mypy.ini`
* Install the mypy plugin: https://github.com/leinardi/mypy-pycharm
* Configure the plugin:
  * Mypy executable path: find it with `where mypy`
  * Arguments `--config-file=mypy.ini`
* This plugin is good to run on individual files using the play button.


## Deleting and creating a Conda environment

To delete, make sure the environment being deleted is not your current environment (just run `deactivate`). Then run 
`conda env remove --name environmentToDelete`.

To create an enviornment from scratch and then export it to a YAML file:

    conda create --name envName python
    pip install whatEverPackage
    pip install packageWithVersion==1.0.42
    conda env export --no-builds --file=my_env.yml

With conda installation, the Apex library is built without the C++ files that are intended be used in backend-op
computations such as fused_adam and fused_layernorm. This is mainly because we are unable to pass the
required input install arguments to the setup file through a conda environment file. By building the library with
these arguments, one could expect further speed-ups in both forward-backward model passes. If you are interested in
installing Apex with these flags, please run the following commands in your shell:

    git clone https://github.com/NVIDIA/apex; cd apex
    git checkout 880ab925bce9f817a93988b021e12db5f67f7787
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

## Conda updates

In order to update the Conda environment, you can go down two routes:
1. You can manually edit the existing `environment.yml` file to force specific (newer) versions of an existing package.
You can do this, for example, to force an update of the `azureml-sdk` and all its contained packages, or `pytorch`
1. Or you can manually add and update packages, and later export the updated environment to a `yml` file.

If you want to take the second route:
1. Use `conda env update -f environment.yml --prune` to refresh if you make changes in environment.yml
1. To update packages use `conda update --all` and `pip-review --local --interactive`

## Using the InnerEye code as a git submodule of your project
You have two options for working with our codebase:
* You can fork the InnerEye-DeepLearning repository, and work off that.
* Or you can create your project that uses the InnerEye-DeepLearning code, and include InnerEye-DeepLearning as a git
submodule.

If you go down the second route, here's the list of files you will need in your project (that's the same as those
given in [this document](building_models.md))
* `environment.yml`: Conda environment with python, pip, pytorch
* `train_variables.yml`: A file similar to `InnerEye\train_variables.yml` containing all your Azure settings
* A folder like `ML` that contains your additional code, and model configurations.
* A file `ML/runner.py` that invokes the InnerEye training runner, but that points the code to your environment and Azure
settings; see the [Building models](building_models.md) instructions for details.

You then need to add the InnerEye code as a git submodule, in folder `innereye-submodule`:
```shell script
git submodule add https://github.com/microsoft/InnerEye-DeepLearning innereye-submodule
```
Then configure your Python IDE to consume *both* your repository root *and* the `innereye-submodule` subfolder as inputs.
In Pycharm, you would do that by going to Settings/Project Structure. Mark your repository root as "Source", and 
`innereye-submodule` as well.
