#!/bin/bash

os_name=$(uname)
if [[ ! $os_name == *"Linux"* ]]; then
    echo "ERROR: cannot run AML environment locking in non-linux environment. Windows users can do this using WSL - https://docs.microsoft.com/en-us/windows/wsl/install"
    exit 1
else
    echo "Starting AML environment locking..."
fi

# get environment name from primary dependencies YAML file
name_line="$(cat primary_deps.yml | grep 'name:')"
IFS=':' read -ra name_arr <<< "$name_line"
env_name="${name_arr[1]}"

# clear old conda envs, create new one
export CONDA_ALWAYS_YES="true"
conda env remove --name ${env_name::-1}
conda env create --file primary_deps.yml

# export new environment to environment.yml
conda env export -n ${env_name::-1} | grep -v "prefix:" > environment.yml
unset CONDA_ALWAYS_YES

# remove python version hash (technically not locked, so still potential for problems here if python secondary deps change)
while IFS='' read -r line; do
    if [[ $line == *"- python="* ]]; then

        IFS='=' read -ra python_arr <<< "$line"
        unset python_arr[-1]
        echo "${python_arr[0]}"="${python_arr[1]}"
    else
        echo "${line}"
    fi
done < environment.yml > environment.yml.tmp
mv environment.yml.tmp environment.yml
