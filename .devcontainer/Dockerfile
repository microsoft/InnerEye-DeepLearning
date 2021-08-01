# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.187.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends make

# [Optional] Uncomment this section to hard code HIML test environment variables into image.
# ENV HIML_RESOURCE_GROUP="<< YOUR_HIML_RESOURCE_GROUP >>"
# ENV HIML_SUBSCRIPTION_ID="<< YOUR_HIML_SUBSCRIPTION_ID >>"
# ENV HIML_WORKSPACE_NAME="<< YOUR_HIML_WORKSPACE_NAME >>"
