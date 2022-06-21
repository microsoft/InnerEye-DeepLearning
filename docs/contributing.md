# Contributing

This document describes guidelines for contributing to the InnerEye-DeepLearning repo.

## Submitting pull requests

- DO submit all changes via pull requests (PRs). They will be reviewed and potentially be merged by maintainers after a peer review that includes at least one of the team members.
- DO give PRs short but descriptive names.
- DO write a useful but brief description of what the PR is for.
- DO refer to any relevant issues and use keywords that automatically close issues when the PR is merged.
- DO ensure each commit successfully builds. The entire PR must pass all checks before it will be merged.
- DO address PR feedback in additional commits instead of amending.
- DO assume that Squash and Merge will be used to merge the commits unless specifically requested otherwise.
- DO NOT submit "work in progress" PRs. A PR should only be submitted when it is considered ready for review.
- DO NOT mix independent and unrelated changes in one PR.

To enable good auto-generated changelogs, we prefix all PR titles with a category string, like "BUG: Out of bounds error when using small images".
Those category prefixes must be in upper case, followed by a colon (`:`). Valid categories are

- `ENH` for enhancements, new capabilities
- `BUG` for bugfixes
- `STYLE` for stylistic changes (for example, refactoring) that does not impact the functionality
- `DOC` for changes to documentation only
- `DEL` for removing something from the codebase

## Coding style

The coding style is enforced via `flake8` and `mypy`. Before pushing any changes to a PR, run both tools on
your dev box:

- `flake8`
- `python mypy_runner.py`

## Unit testing

- DO write unit tests for each new function or class that you add.
- DO extend unit tests for existing functions or classes if you change their core behaviour.
- DO ensure that your tests are designed in a way that they can pass on the local box, even if they are relying on
specific cloud features.
- DO run all unit tests on your dev box before submitting your changes. The test suite is designed to pass completely
also outside of cloud builds.
- DO NOT rely only on the test builds in the cloud. Cloud builds trigger AzureML runs on GPU
machines that have a higher CO2 footprint than your dev box.

## Creating issues

- DO use a descriptive title that identifies the issue or the requested feature.
- DO write a detailed description of the issue or the requested feature.
- DO provide details for issues you create
  - Describe the expected and actual behavior.
  - Provide any relevant exception message.

## Using the hi-ml package

To work on `hi-ml` package at the same time as `InnerEye-DeepLearning`, you can edit `hi-ml` in the git submodule which is automatically cloned as part of the [setup guide](environment.md).

- In the repository root, run `git submodule add https://github.com/microsoft/hi-ml`
- In PyCharm's project browser, mark the folders `hi-ml/hi-ml/src` and `hi-ml/hi-ml-azure/src` as Sources Root
- Remove the entry for the `hi-ml` and `hi-ml-azure` packages from `environment.yml`
- There is already code in `InnerEye.Common.fixed_paths.add_submodules_to_path` that will pick up the submodules and
  add them to `sys.path`.

Once you are done testing your changes:

- Remove the entry for `hi-ml` from `.gitmodules`
- Execute these steps from the repository root:

  ```shell
  git submodule deinit -f hi-ml
  rm -rf hi-ml
  rm -rf .git/modules/hi-ml
  ```

Alternatively, you can consume a developer version of `hi-ml` from `test.pypi`:

- Remove the entry for the `hi-ml` package from `environment.yml`
- Add a section like this to `environment.yml`, to point pip to `test.pypi`, and a specific version of th package:

```yaml
  ...
  - pip:
      - --extra-index-url https://test.pypi.org/simple/
      - hi-ml==0.1.0.post236
      ...
```
