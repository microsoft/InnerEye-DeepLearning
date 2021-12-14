#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
This script is an example of how to use the submit_to_azure_if_needed function from the hi-ml package to run the
main pre-processing function that creates tiles from slides in the TCGA-PRAD dataset. The advantage of using this script
is the ability to submit to a cluster on azureml and to have the output files directly saved as a registered dataset.

To run execute, from inside the pre-processing folder,
python azure_tiles_creation.py --azureml

A json configuration file containing the credentials to the Azure workspace and an environment.yml file are expected
in input.
"""

from pathlib import Path
import sys
import time

current_file = Path(__file__)
radiomics_root = current_file.absolute().parent.parent.parent.parent.parent
sys.path.append(str(radiomics_root))
from health_azure.himl import submit_to_azure_if_needed, DatasetConfig  # noqa

from InnerEye.ML.Histopathology.datasets.tcga_prad_dataset import TcgaPradDataset  # noqa
from InnerEye.ML.Histopathology.preprocessing.create_tiles_dataset import main  # noqa

# Pre-built environment file that contains all the requirements (RadiomicsNN + histo)
# Assuming ENV_NAME is a complete environment, `conda env export -n ENV_NAME -f ENV_NAME.yml` will create the desired file
ENVIRONMENT_FILE = radiomics_root / "envs/innereyeprivatetiles.yml"
INPUT_DATASET_NAME = 'TCGA-PRAD'
OUTPUT_DATASET_NAME = INPUT_DATASET_NAME + '_tiles'
timestr = time.strftime('%Y%m%d-%H%M%S')
folder_name = OUTPUT_DATASET_NAME + '_' + timestr

if __name__ == '__main__':
    print(f"Running {str(current_file)}")
    input_dataset = DatasetConfig(name=INPUT_DATASET_NAME, datastore='innereyedata_premium',
                                  local_folder=Path(f"/tmp/datasets/{INPUT_DATASET_NAME}"), use_mounting=True)
    output_dataset = DatasetConfig(name=OUTPUT_DATASET_NAME, datastore='innereyedata_premium',
                                   local_folder=Path("/datadrive/"), use_mounting=True)
    run_info = submit_to_azure_if_needed(entry_script=current_file,
                                         snapshot_root_directory=radiomics_root,
                                         workspace_config_file=Path("config_westus2.json"),
                                         compute_cluster_name='ND24s',  # 'training-nd24',
                                         conda_environment_file=Path(ENVIRONMENT_FILE),
                                         input_datasets=[input_dataset],
                                         output_datasets=[output_dataset],
                                         )
    input_folder = run_info.input_datasets[0]
    output_folder = Path(run_info.output_datasets[0], folder_name)
    print(f'This will be the final ouput folder {str(output_folder)}')

    if INPUT_DATASET_NAME == 'TCGA-PRAD':
        dataset = TcgaPradDataset(input_folder)
    else:
        raise ValueError(f"Unsupported dataset: {INPUT_DATASET_NAME}")

    main(slides_dataset=dataset,
         root_output_dir=str(output_folder),
         level=1,
         tile_size=224,
         margin=64,
         foreground_threshold=None,
         occupancy_threshold=0.05,
         parallel=True,
         overwrite=False)
