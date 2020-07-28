#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from joblib import Parallel, delayed


def run_mypy(file: str) -> int:
    return subprocess.run(["mypy", "--config=mypy.ini", f"{str(file)}"]).returncode


def main() -> int:
    exclude: List[str] = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted(map(lambda x: x.relative_to(current_dir), Path.cwd().rglob('*.py')))
    files = list(filter(lambda x: not any([str(Path(ele)) in str(x) for ele in exclude]), files))

    return_codes = Parallel(n_jobs=os.cpu_count())(delayed(run_mypy)(file) for file in files)
    if all(v == 0 for v in return_codes):
        return 0
    else:
        sys.stderr.write("mypy failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
