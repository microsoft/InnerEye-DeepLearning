# Building docs for InnerEye-DeepLearning

1. First, make sure you have all the packages necessary for InnerEye.
1. Install pip dependencies from sphinx-docs/requirements.txt.

```shell
pip install -r requirements.txt
```

1. Run `sphinx-build -b html sphinx-docs/source sphinx-docs/build` from the head of the repo. This will create html files under sphinx-docs/build/html.
