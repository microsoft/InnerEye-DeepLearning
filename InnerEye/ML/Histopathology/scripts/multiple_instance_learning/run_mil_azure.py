"""
This script is an example of how to submit the MONAI mil pipeline to AzureML
"""

from pathlib import Path
import sys
import time
import gdown
import os
import torch
import torch.multiprocessing as mp

current_file = Path(__file__)
radiomics_root = current_file.absolute().parent.parent.parent.parent.parent.parent
sys.path.append(str(radiomics_root))
from health_azure.himl import submit_to_azure_if_needed, DatasetConfig  # noqa
from panda_mil_train_evaluate_pytorch_gpu import main_worker, parse_args


# Pre-built environment file that contains all the requirements (InnerEye + Monai)
# Assuming ENV_NAME is a complete environment, `conda env export -n ENV_NAME -f ENV_NAME.yml` will create the desired file
ENVIRONMENT_FILE = "mil_environment2.yml"
INPUT_DATASET_NAME = 'panda'
timestr = time.strftime('%Y%m%d-%H%M%S')

if __name__ == '__main__':
    print(f"Running {str(current_file)}")
    args = parse_args()
    if args.dataset_json is None:
        # download default json datalist
        resource = "https://drive.google.com/uc?id=1L6PtKBlHHyUgTE4rVhRuOLTQKgD4tBRK"
        dst = "./datalist_panda_0.json"
        if not os.path.exists(dst):
            gdown.download(resource, dst, quiet=False)
        args.dataset_json = dst

    input_dataset = DatasetConfig(name=INPUT_DATASET_NAME, datastore='innereyedata_premium',
                                  local_folder=Path(f"/tmp/datasets/{INPUT_DATASET_NAME}"),
                                  target_folder=Path("/tmp/datasets/"),
                                  use_mounting=True)
    run_info = submit_to_azure_if_needed(entry_script=current_file,
                                         snapshot_root_directory=current_file.parent,
                                         workspace_config_file=Path("config.json"),
                                         compute_cluster_name='ND24s',  # 'innereye4cl-a100', 'ND24s', 'training-nd24',
                                         conda_environment_file=Path(ENVIRONMENT_FILE),
                                         input_datasets=[input_dataset],
                                         submit_to_azureml=True,
                                         ignored_folders=["logs"],
                                         num_nodes=args.world_size
                                         )
    input_folder = run_info.input_datasets[0]

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2  # heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)
