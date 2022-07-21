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
    text = filepath.read_text()
    text = text.replace(original_str, replace_str)
    filepath.write_text(text)


if __name__ == '__main__':
    sphinx_root = Path(__file__).absolute().parent
    repository_root = sphinx_root.parent
    docs_root = sphinx_root / "source" / "docs"
    repository_url = "https://github.com/microsoft/InnerEye-DeepLearning"

    # copy README.md and doc files
    shutil.copy(repository_root / "README.md", docs_root)
    shutil.copy(repository_root / "CHANGELOG.md", docs_root)

    # replace links to files in repository with urls
    md_files = docs_root.rglob("*.md")
    for filepath in md_files:
        replace_in_file(filepath, "](/", f"]({repository_url}/blob/main/")
