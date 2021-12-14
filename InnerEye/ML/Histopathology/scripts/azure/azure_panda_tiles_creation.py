#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
This script is an example of how to use the submit_to_azure_if_needed function from the hi-ml package to run the
main pre-processing function that creates tiles from slides in the PANDA dataset. The advantage of using this script
is the ability to submit to a cluster on azureml and to have the output files directly saved as a registered dataset.

This script is specific to PANDA and is kept only for retrocompatibility.
`azure_tiles_creation.py` is the new supported way to process slide datasets.

To run execute, from inside the pre-processing folder,
python azure_tiles_creation.py --azureml

A json configuration file containing the credentials to the Azure workspace and an environment.yml file are expected
in input.

This has been tested on hi-mlv0.1.4.
"""

from pathlib import Path
import sys
import time

current_file = Path(__file__)
radiomics_root = current_file.absolute().parent.parent.parent.parent.parent
sys.path.append(str(radiomics_root))
from health_azure.himl import submit_to_azure_if_needed, DatasetConfig  # noqa
from InnerEye.ML.Histopathology.preprocessing.create_panda_tiles_dataset import main  # noqa

# Pre-built environment file that contains all the requirements (RadiomicsNN + histo)
# Assuming ENV_NAME is a complete environment, `conda env export -n ENV_NAME -f ENV_NAME.yml` will create the desired file
ENVIRONMENT_FILE = radiomics_root.joinpath(Path("/envs/innereyeprivatetiles.yml"))
DATASET_NAME = "PANDA_tiles"
timestr = time.strftime("%Y%m%d-%H%M%S")
folder_name = DATASET_NAME + '_' + timestr

if __name__ == '__main__':
    print(f"Running {str(current_file)}")
    input_dataset = DatasetConfig(name="PANDA", datastore="innereyedatasets", local_folder=Path("/tmp/datasets/PANDA"), use_mounting=True)
    output_dataset = DatasetConfig(name=DATASET_NAME, datastore="innereyedatasets", local_folder=Path("/datadrive/"), use_mounting=True)
    run_info = submit_to_azure_if_needed(entry_script=current_file,
                                         snapshot_root_directory=radiomics_root,
                                         workspace_config_file=Path("config.json"),
                                         compute_cluster_name='training-pr-nc12',  # training-nd24
                                         default_datastore="innereyedatasets",
                                         conda_environment_file=Path(ENVIRONMENT_FILE),
                                         input_datasets=[input_dataset],
                                         output_datasets=[output_dataset],
                                         )
    input_folder = run_info.input_datasets[0]
    output_folder = Path(run_info.output_datasets[0], folder_name)
    print(f'This will be the final ouput folder {str(output_folder)}')

    main(panda_dir=str(input_folder),
         root_output_dir=str(output_folder),
         level=1,
         tile_size=224,
         margin=64,
         occupancy_threshold=0.05,
         parallel=True,
         overwrite=False)
