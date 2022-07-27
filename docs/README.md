# Building docs for InnerEye-DeepLearning

1. First, make sure you have all the packages necessary for InnerEye.
1. Install pip dependencies from sphinx-docs/requirements.txt:

```shell
pip install -r requirements.txt
```

1. Run `make html` from the `docs` folder. This will create html files under docs/build/html.
1. From the `docs/build/html` folder, run `python -m http.server 8080` to host the docs locally.
1. From your browser, navigate to `http://localhost:8080` to view the documentation.
