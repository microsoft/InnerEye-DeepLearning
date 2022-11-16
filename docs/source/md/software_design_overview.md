# Software Development Process

This document provides a high-level overview of the software development process that our team uses for their
repositories (most importantly [InnerEye-DeepLearning](https://github.com/microsoft/InnerEye-DeepLearning) and
[InnerEye-Gateway](https://github.com/microsoft/InnerEye-Gateway)).

## Version Control

Software code versioning is done via GitHub. The code with the highest level of quality control is in the "main" branch.
Ongoing development happens in separate branches. Once development in the branch is finished, a Pull Request (PR)
process is started to integrate the branch into "main".

## Development Process

The design and development of the software in this repository is roughly separated into an **initiation**,
**prototyping**, and a **finalization** phase. The initiation phase can be skipped for minor changes, for example an
update to documentation.

### Initiation

During the initiation phase, the following steps are carried out:

- Collection of a set of requirements.
- Creating a suggested design for the change.
- Review of the design with member of the core team.

The deliverables of this phase are a detailed design of the proposed change in a GitHub Issue or a separate document.

### Prototyping

The engineering owner of the proposed change will create a branch of the current codebase. This branch is separate from
the released (main) branch to not affect any current functionality. In this branch, the engineer will carry out the
changes as proposed in the design document.

The engineer will also add additional software tests at different levels (unit tests, integration tests) as appropriate
for the design. These tests ensure that the proposed functionality will be maintained in the future.

The deliverable of this phase is a branch in the version control system that contains all proposed changes and a set of
software tests.

### Finalization

At this point, the engineering owner of the proposed change has carried out all necessary changes in a branch of the
codebase. They will now initiate a Pull Request process that consists of the following steps:

- The code will be reviewed by at least 2 engineers. Both need to provide their explicit approval before the proposed
  change can be integrated in the "main" branch.
- All unit and integration tests will start.
- All automatic code checks will start. These checks will verify the following:

  - Consistency with static typing rules.
  - Ensure that no passwords or other types of credentials are revealed.
  - Ensure that none of the used third-party libraries contains high severity software vulnerabilities that could affect
    the proposed functionality.

For code to be accepted into the "main" branch, the following criteria need to be satisfied:

- All unit and integration tests pass.
- The code has been reviewed by at least 2 engineers who are members of the core development team. This review will
  also assess non-quantifiable aspects of the proposed change, such as clarity and readability of the code and quality
  of the documentation.
- Any comments that have been added throughout the review process need to be resolved.
- All automated checks pass.

Once all the above criteria are satisfied, the branch will be merged into "main".

## Software Configuration Management

Third party libraries (sometimes called Software of Unknown Provenance, SOUP) are consumed via two
package management systems:

- Conda
- PyPi

Both of those package management systems maintain strict versioning: once a version of a package is published, it
cannot be modified in place. Rather, a new version needs to be released.

Our training and deployment code uses Conda environment files that specify an explicit version of a dependent package to
use (for example, `lightning_bolts==0.4.0`). The Conda environment files are also version controlled in GitHub. Any
change to a version of a third party library will need to be carried out via the same change management process as a code
change, with Pull Request, review, and all tests passing.

The list of third party software is maintained in GitHub in the Conda configuration file, that is `environment.yml` for Linux / AzureML environments and `primary_deps.yml` for all other environments. For example, [this is the latest version of the environment file for the InnerEye-DeepLearning repository](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/environment.yml).

## Defect Handling

The handling of any bugs or defects discovered is done via GitHub Issues.

When an Issue is created, the team members will get notified. Open Issues are kept in a list sorted by priority. For
example, [this is the list for all of the InnerEye-related Issues](https://github.com/orgs/microsoft/projects/320).
Priorities are re-assesed regularly, at least once a week.

The Issue(s) with highest priority are assigned to an engineer. The engineer will then analyze the problem, and
possibly request more information to reproduce the problem. Requesting more information is also handled in the GitHub
Issue. Once the problem is clearly reproduced, the engineer will start to write a fix for the Issue as well as a test that
ensures that the fix does indeed solve the reported problem.

Writing the fix and test will follow the process outlined above in the [Prototyping](#prototyping) and
[Finalization](#finalization) sections. In particular, the fix and test will undergo the Pull Request review process
before they are merged into the main branch.

## Updates to the Development Process

The development process described here is subject to change. Changes will undergo the review process described in this
very document, and will be published on GitHub upon completion.
