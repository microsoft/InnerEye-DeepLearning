# Contributing
This document describes guidelines for contributing to the InnerEye-DeepLearning repo.

## Submitting Pull Requests
- DO submit all changes via pull requests (PRs). They will be reviewed and potentially be merged by maintainers after a peer review that includes at least one of the team members.
- DO give PRs short but descriptive names.
- DO write a useful but brief description of what the PR is for.
- DO refer to any relevant issues and use keywords that automatically close issues when the PR is merged.
- DO ensure each commit successfully builds. The entire PR must pass all checks before it will be merged.
- DO address PR feedback in additional commits instead of amending.
- DO assume that Squash and Merge will be used to merge the commits unless specifically requested otherwise.
- DO NOT submit "work in progress" PRs. A PR should only be submitted when it is considered ready for review.
- DO NOT mix independent and unrelated changes in one PR.

## Coding Style
The coding style is enforced via `flake8` and `mypy`. Before pushing any changes to a PR, run both tools on
your dev box:
* `flake8`
* `python mypy_runner.py`

## Unit testing
- DO write unit tests for each new function or class that you add.
- DO extend unit tests for existing functions or classes if you change their core behaviour.
- DO ensure that your tests are designed in a way that they can pass on the local box, even if they are relying on
specific cloud features.
- DO run all unit tests on your dev box before submitting your changes. The test suite is designed to pass completely
also outside of cloud builds.
- DO NOT rely only on the test builds in the cloud. Cloud builds trigger AzureML runs on GPU 
machines that have a higher CO2 footprint than your dev box.

## Creating Issues
- DO use a descriptive title that identifies the issue or the requested feature.
- DO write a detailed description of the issue or the requested feature.
- DO provide details for issues you create:
    - Describe the expected and actual behavior.
    - Provide any relevant exception message.
DO subscribe to notifications for created issues in case there are any follow-up questions.
