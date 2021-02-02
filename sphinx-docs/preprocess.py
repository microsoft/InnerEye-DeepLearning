#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from pathlib import Path


def replace_in_file(filepath: Path, original_str: str, replace_str: str) -> None:
    """
    Replace all occurences of the original_str with replace_str in the file provided.
    """
    with filepath.open('r') as file:
        text = file.read()
    text = text.replace(original_str, replace_str)
    with filepath.open('w') as file:
        file.write(text)


if __name__ == '__main__':
    sphinx_root = Path(__file__).absolute().parent
    repository_root = sphinx_root.parent
    md_root = sphinx_root / "source/md"
    repository_url = "https://github.com/microsoft/InnerEye-DeepLearning"

    # Create directories source/md and source/md/docs where files will be copied to
    if md_root.exists():
        shutil.rmtree(md_root)
    md_root.mkdir()

    # copy README.md and doc files
    shutil.copyfile(repository_root / "README.md", md_root / "README.md")
    shutil.copytree(repository_root / "docs", md_root / "docs")

    # replace links to files in repository with urls
    md_file_list = md_root.rglob("*.md")
    for filepath in md_file_list:
        replace_in_file(filepath, "](/", f"]({repository_url}/blob/main/")
