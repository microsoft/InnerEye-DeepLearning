# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.187.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

RUN apt-get update
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install --no-install-recommends make
RUN apt-get -y install gcc git-lfs python3-dev build-essential

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update --name base --file /tmp/conda-tmp/environment.yml; fi
RUN rm -rf /tmp/conda-tmp
# N.B. Despite the previous lines I still needed to run
#     conda update --name base conda --channel anaconda
#     conda env create --file environment.yml
# manually in the container after it started. TODO: Fix that.

# [Optional] Uncomment this section to hard code HIML test environment variables into image.
# ENV HIML_RESOURCE_GROUP="<< YOUR_HIML_RESOURCE_GROUP >>"
# ENV HIML_SUBSCRIPTION_ID="<< YOUR_HIML_SUBSCRIPTION_ID >>"
# ENV HIML_WORKSPACE_NAME="<< YOUR_HIML_WORKSPACE_NAME >>"
