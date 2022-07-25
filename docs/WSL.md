# How to use the Windows Subsystem for Linux (WSL2) for development

We are aware of two issues with running our toolbox on Windows:

- Conda and miniconda can be rather temperamental: Environment creation can fail with package conflict errors of unclear
  origin, or internal conda errors.
- Some features of PyTorch are not supported, or not well supported, on Windows.

If you are facing issue of the above kind on a Windows machine, we would highly recommend working with the Windows
Subsystem for Linux (WSL2) or a plain Ubuntu Linux box.

## Enable CUDA in WSL2

If you are running a Windows box with a GPU, please follow the documentation
[here](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) to access the GPU from within WSL2.

You can also find a video walkthrough of WSL2+CUDA installation
here: https://channel9.msdn.com/Shows/Tabs-vs-Spaces/GPU-Accelerated-Machine-Learning-with-WSL-2

## Install WSL2

Requirements: Windows 10 version 2004 or higher

To use the commandline setup, please first install
[winget via the appxbundle](https://github.com/microsoft/winget-cli/releases).

Optionally, restart your machine.

In PowerShell as Administrator type:
```
wsl --install
```

Then, restart your machine one more time.

Since it is possible to choose the version of WSL that a particular distribution is running, once you have WSL2
installed, ensure that your distribution is running on top of WSL2 by executing
`wsl --list --verbose`
If all is good, the output should look like this:

```
$> wsl --list -v
  NAME            STATE           VERSION
* Ubuntu-20.04    Running         2
```

Note the "2" in the "Version" column.

The instructions are [here](https://docs.microsoft.com/en-us/windows/wsl/install), but summarized in copy/paste-able form above. Optionally, you can install via the UI, pick Ubuntu version 20.04 LTS as your distribution.

Then, you can start the Ubuntu either directly from the Start menu, or via the WindowsTerminal app.

## Install git and Anaconda

Start the Windows Terminal app, create an Ubuntu tab. In the shell, run the following commands:

- `sudo apt update`
- `sudo apt install git git-lfs python-dev build-essential`
- `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- `sh Miniconda3-latest-Linux-x86_64.sh`
- Close your WSL shell and re-start it
- Clone repo or access your repos via /mnt/c/...
- Create conda environment: `conda env create --file environment.yml`
- Clean your pyc files (in case you have some left from Windows):

```
find * -name '*.pyc' | xargs -d'\n' rm`
```

## Configure PyCharm

- https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html
- You might need to reset all your firewall settings to make the debugger work with PyCharm. This can be done with these
  PowerShell commands (as Administrator):

```
$myIp = (Ubuntu2004 run "cat /etc/resolv.conf | grep nameserver | cut -d' ' -f2")
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -LocalAddress $myIp -Action Allow
```

- Then (re)start PyCharm. If asked whether to give it permission to communicate over domain, private and public
  networks, make sure all three are ticked.
- If you are still struggling with the firewall rules, consider removing all your current firewall rules, by running
  `Remove-NetFirewallRule` in the PowerShell. WARNING: This will remove all your present firewall rules, and you may
  need to repeat the firewall setup for other programs that you have installed!

## Configure VSCode

- https://code.visualstudio.com/docs/remote/wsl
