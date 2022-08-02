# Microsoft Research Cambridge InnerEyeDeepLearning for Medical Image Analysis

## Consuming the InnerEye package

* You need to have a Conda installation on your machine.
* Create a Conda environment file `environment.yml` in your source code with this contents:

```yaml
name: MyEnv
channels:
  - defaults
  - pytorch
dependencies:
  - pip=20.0.2
  - python=3.7.3
  - pytorch=1.3.0
  - pip:
      - innereye
```

* Create a conda environment: `conda env create --file environment.yml`
* Activate the environment: `conda activate MyEnv`
