#!/bin/bash

# create conda environment
export CONDA_ALWAYS_YES="true"
conda env create --file primary_deps.yml
name_line="$(cat primary_deps.yml | grep 'name:')"
IFS=':' read -ra name_arr <<< "$name_line"
env_name="${name_arr[1]}"
conda env export -n ${env_name::-1} | grep -v "prefix:" > environment.yml
unset CONDA_ALWAYS_YES

# remove python version hash
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

echo Activate your environment by running the following command:
echo "      conda activate" ${env_name}
