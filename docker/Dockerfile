FROM continuumio/miniconda3:latest

MAINTAINER antonsc@microsoft.com

RUN apt-get -y update

# Installing psutil requires gcc. Install sudo to help with permissions problems
RUN apt-get -y install \
    sudo

RUN apt-get clean

# Ensure everyone has write permissions, because we need to update the environment in the build pipeline
RUN chmod -R a+w /opt/conda/
