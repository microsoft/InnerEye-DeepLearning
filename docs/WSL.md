# How to use the Windows Subsystem for Linux (WSL2) for development

We are aware of two issues with running our toolbox on Windows:
- Conda and miniconda can be rather temperamental: Environment creation can fail with package conflict errors
of unclear origin, or internal conda errors.
- Some features of PyTorch are not supported, or not well supported, on Windows.

If you are facing issue of the above kind on a Windows machine, we would highly recommend working with the
Windows Subsystem for Linux (WSL2) or a plain Ubuntu Linux box.

## Enable CUDA in WSL2
If you are running a Windows box with a GPU, please follow the documentation 
[here](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) to access the GPU from within WSL2.

You can also find a video walkthrough of WSL2+CUDA installation here: https://channel9.msdn.com/Shows/Tabs-vs-Spaces/GPU-Accelerated-Machine-Learning-with-WSL-2

## Install WSL2

Requirements: Windows 10 version 2004 or higher

The instructions are [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10), but summarized in
copy/paste-able form below. When installing via the UI, pick Ubuntu version 20.04 LTS as your distribution.

To use the commandline setup, please first install 
[winget via the appxbundle](https://github.com/microsoft/winget-cli/releases).

Then, in PowerShell as Administrator:
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
winget install ubuntu --version 20.04
wsl --list --verbose
wsl --set-version Ubuntu-20.04 2
wsl --set-default-version 2
winget install Microsoft.WindowsTerminal
```

Remember to restart your machine if you were doing a fresh installation of WSL 2 before trying further steps. 

Since it is possible to choose the version of WSL that a particular distribution is running, 
once you have WSL2 installed, ensure that your distribution is running on top of WSL2 by executing  
`wsl --list --verbose`  
If all is good, the output should look like this:  
```
$> wsl --list -v
  NAME            STATE           VERSION
* Ubuntu-20.04    Running         2
```
Note the "2" in Version column.


## Install git and Anaconda

Start the Windows Terminal app, create an Ubuntu tab, and inside that:
- `sudo apt update`
- `sudo apt install git`
- `sudo apt-get install git-lfs`
- `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- `sh Miniconda3-latest-Linux-x86_64.sh`
- `sudo apt-get install python-dev`
- `sudo apt-get install build-essential`
- Clone repo or access your repos via /mnt/c/...
- Create conda environment: `conda env create --file environment.yml`
- Create a file in the top level directory of the repository, named `InnerEyeTestVariables.txt`, with one line:
```
APPLICATION_KEY=<app key for your AML workspace>
```
This will enable you to run tests that require authentication to Azure.
- Clean your pyc files (in case you have some left from Windows):
```
find * -name '*.pyc' | xargs -d'\n' rm`
```

## Configure PyCharm

- https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html
- You might need to reset all your firewall settings to make the debugger work with PyCharm. This can be done with these PowerShell commands (as Administrator):
```
Remove-NetFirewallRule
$myIp = (Ubuntu1804 run "cat /etc/resolv.conf | grep nameserver | cut -d' ' -f2")
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -LocalAddress $myIp -Action Allow
```
- Then (re)start PyCharm. If asked whether to give it permission to communicate over domain, private and public networks, make sure all three are ticked.

## Configure VSCode
- https://code.visualstudio.com/docs/remote/wsl
