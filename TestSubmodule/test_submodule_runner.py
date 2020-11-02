#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys
from pathlib import Path


# This file here mimics how the InnerEye code would be used as a git submodule. The test script will
# copy the InnerEye code to a folder called Submodule. The test will then invoke the present file as a runner,
# and train a model in AzureML.

repository_root = Path(__file__).absolute().parent.parent


def add_package_to_sys_path_if_needed() -> None:
    """
    Checks if the Python paths in sys.path already contain the /Submodule folder. If not, add it.
    """
    is_package_in_path = False
    innereye_submodule_folder = repository_root / "Submodule"
    for path_str in sys.path:
        path = Path(path_str)
        if path == innereye_submodule_folder:
            is_package_in_path = True
            break
    if not is_package_in_path:
        print(f"Adding {innereye_submodule_folder} to sys.path")
        sys.path.append(str(innereye_submodule_folder))


def main() -> None:
    try:
        from InnerEye import ML  # noqa: 411
    except:
        add_package_to_sys_path_if_needed()

    from InnerEye.ML import runner
    from InnerEye.Common import fixed_paths
    print(f"Repository root: {repository_root}")
    runner.run(project_root=repository_root,
               yaml_config_file=fixed_paths.SETTINGS_YAML_FILE,
               post_cross_validation_hook=None)


if __name__ == '__main__':
    main()
