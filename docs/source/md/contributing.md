# Contributing

This document contains detailed guidelines for contributing to the InnerEye-DeepLearning repo.

For the overarching software development process, refer to the [process description](software_development_process.md).

## Submitting pull requests

- DO submit all changes via pull requests (PRs). They will be reviewed and potentially be merged by maintainers after
  a peer review that includes at least one of the team members.
- DO NOT mix independent and unrelated changes in one PR. PRs should implement one change, and hence be small. If
  in doubt, err on the side of making the PR too small, rather than too big. It reduces the chance for you as the
  author to overlook issues. Small PRs are easier to test and easier to review.
- DO give PRs short but descriptive names.
- DO write a useful but brief description of what the PR is for.
- DO refer to any relevant issues and use keywords that automatically close issues when the PR is merged.
- DO ensure each commit successfully builds. The entire PR must pass all checks before it will be merged.
- DO link the correct Github issue.
- DO address PR feedback in additional commits instead of amending existing commits.
- DO NOT add any changes that go beyond what is requested by the reviewers.
- DO assume that Squash and Merge will be used to merge the commits unless specifically requested otherwise.

To enable good auto-generated changelogs, we prefix all PR titles with a category string, like
"BUG: Out of bounds error when using small images".
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
- DO add tests for each bug you fix. When fixing a bug, the suggested workflow is
  to first write a unit test that shows the invalid behaviour, and only then start to code up the fix.
- DO run all unit tests on your dev box before submitting your changes. The test suite is designed to pass completely
  also outside of cloud builds. If you are not
  a member of the core InnerEye team, note that you may not be able to run some of the unit tests that access the team's
  AzureML workspace. A member of the InnerEye team will be happy to assist then.
- DO NOT rely only on the test builds in the cloud. Cloud builds trigger AzureML runs on GPU
  machines that have a higher CO2 footprint than your dev box.

More details, in particular on tests that require GPUs, can be found in the [testing documentation](testing.md).

## Creating issues

- DO use a descriptive title that identifies the issue or the requested feature.
- DO write a detailed description of the issue or the requested feature.
- DO provide details for issues you create
  - Describe the expected and actual behavior.
  - Provide any relevant exception message.
