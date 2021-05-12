## Suggested Workflow for Pull Requests

* Pull Requests (PRs) should implement one change, and hence be small. If in doubt, err on the side of making the PR
too small, rather than too big. It reduces the chance for you as the author to overlook issues. Small PRs are easier
to test and easier to review.
* Ensure that the code you add or change is covered by unit tests. When fixing a bug, the suggested workflow is
to first write a unit test that shows the invalid behaviour, and only then start to code up the fix.
* It is best to run all unit tests on your dev box, and check that they pass, before publishing the PR. If you are not
a member of the core InnerEye team, note that you may not be able to run some of the unit tests that access the team's
AzureML workspace. A member of the InnerEye team will be happy to assist then.
* Before publishing your PR, please run PyCharm's code cleanup tools. You can either do that after editing your file,
by pressing Ctrl+Alt+L, or selecting "Reformat code" in the context menu of the file(s) in the project explorer window.
Alternatively, you should tick all of "Reformat code", "Rearrange code", "Optimize imports", "Cleanup", "Scan with mypy"
in the PyCharm version control check-in dialog.
* Ensure that you modified [CHANGELOG.md](../CHANGELOG.md) and described your PR there.
* Only publish your PR for review once you have a build that is passing. You can make use of the "Create as Draft"
feature of GitHub.
* Link the correct Github issue.
* Once you have obtained approval for your change, do not add any changes that go beyond what is requested by the
reviewers. Any additional pushes to the branch will invalidate the approvals you have obtained.
