FROM continuumio/miniconda3:latest

MAINTAINER antonsc@microsoft.com

# Environment file is copied to this folder by the build definition
COPY ./environment.yml /tmp

RUN conda env create --file /tmp/environment.yml
