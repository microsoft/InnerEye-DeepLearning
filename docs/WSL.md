# How to use WSL2 for development

## Install WSL2

Requirements: Windows 10 version 2004 or higher
https://docs.microsoft.com/en-us/windows/wsl/install-win10

Install winget from the appxbundle at https://github.com/microsoft/winget-cli/releases

On PowerShell as Administrator:
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl --set-default-version 2
winget install ubuntu --version 20.04
wsl --list --verbose
wsl --set-version <distribution name> <versionNumber>
wsl --set-default-version 2
winget install Microsoft.WindowsTerminal
```

## Install git and Anaconda

Start the Windows Terminal app, create an Ubuntu tab, and inside that:
- sudo apt update
- sudo apt install git
- sudo apt-get install git-lfs
- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- sh Miniconda3-latest-Linux-x86_64.sh
- sudo apt-get install python-dev
- sudo apt-get install build-essential
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
