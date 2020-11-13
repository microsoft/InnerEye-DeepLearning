# Releases

The InnerEye toolbox is in an early stage, where many of its inner workings are changing rapidly. However, the
purely config-driven approach to model building should remain stable. That is, you can expect backwards
compatibility if you are building models by creating configuration files and changing the fields of the classes 
that define, say, a segmentation model. The same goes for all Azure-related configuration options.
If your code relies on specific functions inside the InnerEye code base, you should expect that this can change.

The current InnerEye codebase is not published as a Python package, and hence does not have implicit version numbers.
We are applying tagging instead, with increases corresponding to what otherwise would be major/minor versions. 

Please refer to the [Changelog](../CHANGELOG.md) for an overview of recent changes.
