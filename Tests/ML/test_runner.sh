#!/usr/bin/env bash

# Assumes all tests if no argument was given
# cpu: test everything that does need a gpu
# gpu: test everything that needs a gpu
# all: run all tests

if [[ $# -eq 0 ]]
then
    folder=../tests
    echo running all tests
else
    echo unknown arg, exiting
    exit -1
fi

cd ../ML

python -m pytest -v -s --cov-report term-missing --cov=. ${folder}
