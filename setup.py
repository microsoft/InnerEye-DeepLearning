#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from typing import List

import ruamel.yaml
import setuptools
from ruamel.yaml.comments import CommentedMap

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import namespace_to_path
from InnerEye.Common.fixed_paths import INNEREYE_PACKAGE_NAME, INNEREYE_PACKAGE_ROOT

ML_NAMESPACE = "InnerEye.ML"

ENVIRONMENT_PATH_ON_PACKAGE = str(Path(INNEREYE_PACKAGE_NAME))

package_root = Path.cwd()

long_description = (package_root / INNEREYE_PACKAGE_NAME / "README.md").read_text()

BUILDID_ENV = "BUILDID"
build_id = os.getenv(BUILDID_ENV, "1")

SOURCEBRANCHNAME_ENV = "SOURCEBRANCHNAME"
build_branchname = os.getenv(SOURCEBRANCHNAME_ENV, "NOT_MASTER")

IS_DEV_PACKAGE_ENV = "IS_DEV_PACKAGE"
is_dev_package = os.getenv(IS_DEV_PACKAGE_ENV, False) == "True"

# Determine package version based on the source branch name being built from.
package_minor_version = 1 if build_branchname == "master" else 0

# The full version of the package that we are creating is later needed for running pytest,
# because we want to install the newly created package from the feed.
package_version = f"0.{package_minor_version}.{build_id}"
Path("package_version.txt").write_text(package_version)


def _get_innereye_packages() -> List[str]:
    return [x for x in setuptools.find_namespace_packages(str(package_root)) if x.startswith(INNEREYE_PACKAGE_NAME)]


def _get_source_packages(exclusions: List[str] = None) -> List[str]:
    exclusions = [] if not exclusions else exclusions
    return [x for x in _get_innereye_packages() if not any([e in x for e in exclusions])]


def _get_init_py_path(namespace: str) -> Path:
    return namespace_to_path(namespace) / "__init__.py"


def _namespace_contains_init_py(namespace: str) -> bool:
    return _get_init_py_path(namespace).exists()


def _pre_process_packages() -> List[str]:
    shutil.copy(fixed_paths.ENVIRONMENT_YAML_FILE_NAME, ENVIRONMENT_PATH_ON_PACKAGE)

    packages_to_preprocess = [x for x in _get_innereye_packages() if not _namespace_contains_init_py(x)]
    for x in packages_to_preprocess:
        _get_init_py_path(x).touch()

    print("Pre-processed packages: ", packages_to_preprocess)
    return packages_to_preprocess


def _post_process_packages(packages: List[str]) -> None:
    for x in packages:
        _get_init_py_path(x).unlink()

    (Path(INNEREYE_PACKAGE_NAME) / fixed_paths.ENVIRONMENT_YAML_FILE_NAME).unlink()
    print("Post-processed packages: ", packages)


def _get_child_package(root: str, child: str) -> str:
    return f"{root}.{child}"


published_package_name = "innereye"

package_data = {
    INNEREYE_PACKAGE_NAME: [fixed_paths.ENVIRONMENT_YAML_FILE_NAME, fixed_paths.VISUALIZATION_NOTEBOOK_PATH]
}

for file in package_data[INNEREYE_PACKAGE_NAME]:
    if not os.path.exists(os.path.join(str(package_root), file)) \
            and not os.path.exists(os.path.join(str(package_root), INNEREYE_PACKAGE_NAME, file)):
        raise FileNotFoundError(f"File {file} in package data does not exist")

with open(package_root / fixed_paths.ENVIRONMENT_YAML_FILE_NAME, "r") as env_file:
    yaml_contents = ruamel.yaml.round_trip_load(env_file)
env_dependencies = yaml_contents["dependencies"]
pip_list = None
for d in env_dependencies:
    if isinstance(d, CommentedMap) and "pip" in d:
        pip_list = list(d["pip"])
if not pip_list:
    raise ValueError("Expected there is a 'pip' section in the environment file?")

git_prefix = "git+"
required_packages = []
for requirements_line in pip_list:
    if requirements_line.startswith(git_prefix):
        print(f"Pypi does not allow to add a git references as a dependency. Skipping: {requirements_line}")
    elif not requirements_line.startswith("-"):
        required_packages.append(requirements_line)

if is_dev_package:
    published_package_name += "-dev"
    package_data[INNEREYE_PACKAGE_NAME] += [
        fixed_paths.SETTINGS_YAML_FILE_NAME,
    ]
    print("\n ***** NOTE: This package is built for development purpose only. DO NOT RELEASE THIS! *****")
    print(f"\n ***** Will install dev package data: {package_data} *****\n")

package_data[INNEREYE_PACKAGE_NAME] += [
    str(INNEREYE_PACKAGE_ROOT / r"ML/reports/segmentation_report.ipynb"),
    str(INNEREYE_PACKAGE_ROOT / r"ML/reports/classification_report.ipynb")
]

pre_processed_packages = _pre_process_packages()
try:
    setuptools.setup(
        name=published_package_name,
        version=package_version,
        author="Microsoft Research Cambridge InnerEye Team ",
        author_email="innereyedev@microsoft.com",
        description="Contains code for working with medical images.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=required_packages,
        # We are making heavy use of dataclasses, which are not yet supported in Python 3.6
        python_requires='>=3.7',
        package_data=package_data,
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Operating System :: Linux",
            "Operating System :: Windows",
        ],
        packages=[INNEREYE_PACKAGE_NAME, *_get_source_packages()]
    )
finally:
    _post_process_packages(pre_processed_packages)
