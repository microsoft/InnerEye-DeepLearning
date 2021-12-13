FROM continuumio/miniconda3:latest

MAINTAINER antonsc@microsoft.com

RUN apt-get -y update

# Installing psutil requires gcc. Install sudo to help with permissions problems
RUN apt-get -y install \
    gcc \
    sudo

# Environment file is copied to this folder by the build definition
COPY ./environment.yml /tmp

RUN conda env create --file /tmp/environment.yml \
    && conda clean --all \
    # Ensure everyone has write permissions, because we need to update the environment in the build pipeline
    && chmod -R a+w /opt/conda/envs
