#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import shutil
from pathlib import Path


def replace_in_file(filepath: Path, original_str: str, replace_str: str) -> None:
    """
    Replace all occurences of the original_str with replace_str in the file provided.
    """
    text = filepath.read_text()
    text = text.replace(original_str, replace_str)
    filepath.write_text(text)


if __name__ == '__main__':
    sphinx_root = Path(__file__).absolute().parent
    repository_root = sphinx_root.parent
    markdown_root = sphinx_root / "source" / "md"
    repository_url = "https://github.com/microsoft/InnerEye-DeepLearning"

    # Create directories source/md and source/md/docs where files will be copied to
    if markdown_root.exists():
        shutil.rmtree(markdown_root)
    markdown_root.mkdir()

    # Copy all markdown files to the markdown directory
    logging.info("Copying markdown files to {}".format(markdown_root / "docs"))
    shutil.copytree(repository_root / "docs", markdown_root / "docs")
    shutil.copy(repository_root / "README.md", markdown_root / "docs")
    shutil.copy(repository_root / "CHANGELOG.md", markdown_root / "docs")

    # replace links to files in repository with urls
    md_files = markdown_root.rglob("*.md")
    for filepath in md_files:
        replace_in_file(filepath, "](/", f"]({repository_url}/blob/main/")
