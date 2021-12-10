FROM continuumio/miniconda3:latest

MAINTAINER antonsc@microsoft.com

RUN apt-get -y update

# Installing psutil requires gcc
RUN apt-get -y install gcc

# Environment file is copied to this folder by the build definition
COPY ./environment.yml /tmp

RUN conda env create --file /tmp/environment.yml
RUN chmod a+w /opt/conda/envs
