#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
current_file = Path(__file__)
root_dir = current_file.parent

DATASETPATH = ""
ENVIRONMNENT = str(root_dir / "environment.yml")
WORKSPACECONFIG = ""
CLUSTER = ""
DATASTORE = ""
TRAINEDSTYLEGAN2WEIGHTS = str(root_dir / 'assets' / 'epoch=999-step=168999.ckpt')
GENSCANSPATH = str(root_dir / 'assets' / 'extracted_scans')
