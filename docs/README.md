# Building docs for InnerEye-DeepLearning

1. First, make sure you have set up your conda environment as described in the [Quick Setup Guide](../README.md#quick-setup).
2. Run `make html` from the `docs` folder. This will create html files under docs/build/html.
3. From the `docs/build/html` folder, run `python -m http.server 8080` to host the docs locally.
4. From your browser, navigate to `http://localhost:8080` to view the documentation.
