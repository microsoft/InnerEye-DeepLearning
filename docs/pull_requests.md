## How to do Pull Requests

* Pull Requests (PRs) should implement one change, and hence be small. If in doubt, err on the side of making the PR
too small, rather than too big. It reduces the chance for you as the author to overlook issues. Small PRs are easier
to test and easier to review.
* Before publishing your PR, please run PyCharm's code cleanup tools. You can either do that after editing your file,
by pressing Ctrl+Alt+L, or selecting "Reformat code" in the context menu of the file(s) in the project explorer window.
Alternatively, you should tick all of "Reformat code", "Rearrange code", "Optimize imports", "Cleanup", "Scan with mypy"
in the PyCharm version control check-in dialog.
* Link the correct Github issue.
* Only publish your PR for review once you have a build that is passing. You can make use of the "Create as Draft"
feature of GitHub.
* Once you have obtained approval for your change, do not add any changes that go beyond what is requested by the
reviewers. Any additional pushes to the branch will invalidate the approvals you have obtained.
