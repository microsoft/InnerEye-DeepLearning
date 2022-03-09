# Setting up your environment

## Prerequisites

In order to work with the solution, your OS environment will need [git](https://git-scm.com/) and [git lfs](https://git-lfs.github.com/) installed. Depending on the OS that you are running the installation instructions may vary. Please refer to respective documentation sections on the tools' websites for detailed instructions. 

We recommend using PyCharm or VSCode as the Python editor. 

You have two options for working with our codebase:
* You can fork the InnerEye-DeepLearning repository, and work off that. We recommend that because it is easiest to set up.
* Or you can create your project that uses the InnerEye-DeepLearning code, and include InnerEye-DeepLearning as a git
submodule. We only recommended that if you are very handy with Python. More details about this option 
[are here](innereye_as_submodule.md).

## Windows Subsystem for Linux Setup
When developing on a Windows machine, we recommend using [the Windows Subsystem for Linux, WSL2](https://docs.microsoft.com/en-us/windows/wsl/about).
That's because PyTorch has better support for Linux.  If you want to use WSL2, please follow 
[these instructions](/docs/WSL.md) , that correspond to the manual installation in the official docs.

## Installing Conda or Miniconda
You can skip this step if you have installed WSL as per the previous item.

Download a Conda or Miniconda [installer for your platform](https://docs.conda.io/en/latest/miniconda.html)
and run it.

## Creating a Conda environment
Note that in order to create the Conda environment you will need to have build tools installed on your machine. If you are running Windows, they should be already installed with Conda distribution.   

You can install build tools on Ubuntu (and Debian-based distributions) by running  
`sudo apt-get install build-essential`  
If you are running CentOS/RHEL distributions, you can install the build tools by running  
`yum install gcc gcc-c++ kernel-devel make`

Start the `conda` prompt for your platform. In that prompt, navigate to your repository root and run
`conda env create --file environment.yml`

## Using GPU locally

It is possible to run the training process on a local machine. It will not be as performant as using a GPU cluster that Azure ML offers and you will not be able to take advantage of other Azure ML features such as comparing run results, creating snapshots for repeatable machine learning experiments or keeping history of experiment runs. At the same time it could be useful to experiment with code or troubleshoot things locally. 

The SDK uses PyTorch to compose and run DNN computations. PyTorch can leverage the underlying GPU via NVidia CUDA technology, which accelerates computations dramatically. 

In order to enable PyTorch to use CUDA, you need to make sure that you have  
1. Compatible graphics card with CUDA compute capability of at least 3.0 (at the moment of writing). You can check compatibility list here: https://developer.nvidia.com/cuda-gpus  
1. Recent NVidia drivers installed

A quick way to check if PyTorch can use the underlying GPU for computation is to run the following line from your conda environment with all InnerEye packages installed:  
`python -c 'import torch; print(torch.cuda.is_available())'`  
It will output `True` if CUDA computation is available and `False` if it's not.

Some tips for installing NVidia drivers below:

### Windows
You can download NVidia drivers for your graphics card from https://www.nvidia.com/download/index.aspx as a Windows *.exe* file and install them this way. 

### WSL
Microsoft provides GPU support via WSL starting WSL 2.0. 

You can find more details on WSL in our separate [WSL section](WSL.md).

### Linux
The exact instructions for driver installation will differ depending on the Linux distribution. Generally, you should first run the `nvidia-smi` tool to see if you have NVidia drivers installed. This tool is installed together with NVidia drivers and if your system can not find it, it may mean that the drivers are not installed. A sample output of NVidia SMI tool may look like this:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 0000027F:00:00.0 Off |                    0 |
| N/A   50C    P0    60W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

In this case we can see that the system has access to a Tesla K80 GPU and is running driver version 450.51.06

If the driver is not available, you can try the following to install:

#### Ubuntu
1. Run  
`ubuntu-drivers devices`  
to see what drivers are available (you may need to install the tool via `sudo apt-get install ubuntu-drivers-common` and update the package database via `sudo apt update`). You should see an output like this:  
```
...
vendor   : NVIDIA Corporation
model    : GK210GL [Tesla K80]
driver   : nvidia-driver-450-server - distro non-free recommended
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-440-server - distro non-free
driver   : nvidia-driver-435 - distro non-free
driver   : nvidia-driver-450 - distro non-free
driver   : nvidia-driver-390 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```
2. Run  
`sudo apt install nvidia-driver-450-server`  
(or whichever is the recommended in your case)
3. Reboot your system

At this point you should be able to run the `nvidia-smi` tool and PyTorch should be able to communicate with the GPU

#### CentOS/RHEL
1. Add NVidia repository to your config manager  
`sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo` (if you are running RHEL8, otherwise you can get the URL for your repo from here: https://developer.download.nvidia.com/compute/cuda/repos/)
2. Clean repository cache via  
`sudo dnf clean all`
3. Install drivers  
`sudo dnf -y module install nvidia-driver:latest-dkms`  
4. Reboot your system

At this point you should be able to run the `nvidia-smi` tool and PyTorch should be able to communicate with the GPU

You can find instructions for other Linux distributions on NVidia website: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 

# More Details for Tool Setup
The following steps describe how to set up specific tools. You can execute most of those at a later
point, if you want to dig deeper into the code.

## VSCode

([VSCode](https://code.visualstudio.com/) for example)

## Conda
-  `conda env create -f environment.yml`

## Conda updates

In order to update the Conda environment, you can go down two routes:
1. You can manually edit the existing `environment.yml` file to force specific (newer) versions of an existing package.
You can do this, for example, to force an update of the `azureml-sdk` and all its contained packages, or `pytorch`
1. Or you can manually add and update packages, and later export the updated environment to a `yml` file.

If you want to take the second route:
1. Use `conda env update -f environment.yml --prune` to refresh if you make changes in environment.yml
1. To update packages use `conda update --all` and `pip-review --local --interactive`


## Using the hi-ml package

To work on `hi-ml` package at the same time as `InnerEye-DeepLearning`, it can help to add the `hi-ml` package
as a submodule, rather than a package from pypi. Any change to the package will require a full new docker image build,
and that costs 20min per run.

* In the repository root, run `git submodule add https://github.com/microsoft/hi-ml`
* In PyCharm's project browser, mark the folders `hi-ml/hi-ml/src` and `hi-ml/hi-ml-azure/src` as Sources Root
* Remove the entry for the `hi-ml` and `hi-ml-azure` packages from `environment.yml`
* There is already code in `InnerEye.Common.fixed_paths.add_submodules_to_path` that will pick up the submodules and
  add them to `sys.path`.

Once you are done testing your changes:
* Remove the entry for `hi-ml` from `.gitmodules` 
* Execute these steps from the repository root:
```shell
git submodule deinit -f hi-ml
rm -rf hi-ml
rm -rf .git/modules/hi-ml
```

Alternatively, you can consume a developer version of `hi-ml` from `test.pypi`:
* Remove the entry for the `hi-ml` package from `environment.yml`
* Add a section like this to `environment.yml`, to point pip to `test.pypi`, and a specific version of th package:
```
  ...
  - pip:
      - --extra-index-url https://test.pypi.org/simple/
      - hi-ml==0.1.0.post236
      ...
```
