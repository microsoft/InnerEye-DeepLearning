#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from InnerEye.Common import fixed_paths, fixed_paths_for_tests


def test_invalid_python_packages() -> None:
    """
    Test if the Python environments that we use do not contain packages that might cause trouble with
    SSL connections.
    """
    packages_to_avoid = [
        "ca-certificates",
        "openssl",
        "ndg-httpsclient",
        "pyopenssl",
        "urllib3"
        # Windows-specific packages
        "certifi"
        "icc_rt"
        "vc"
        "vs2015_runtime"
        "wincertstore"
        "pypiwin32"
        "pywin32"
        "pywinpty"
    ]

    def check_file(file: Path) -> None:
        with file.open("r") as f:
            for line in f:
                for package in packages_to_avoid:
                    assert package not in line, "Package {package} should be avoided, but found in: {line}"

    print("Full set of packages that will cause problems:")
    for package in packages_to_avoid:
        print("- {}".format(package))
    check_file(fixed_paths_for_tests.tests_root_directory().parent / fixed_paths.ENVIRONMENT_YAML_FILE_NAME)
