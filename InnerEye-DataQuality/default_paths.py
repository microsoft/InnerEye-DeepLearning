#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.resolve()
INNEREYE_DQ_DIR = PROJECT_ROOT_DIR / "InnerEyeDataQuality"
CIFAR10_ROOT_DIR = PROJECT_ROOT_DIR / "data" / "CIFAR10"

EXPERIMENT_DIR = PROJECT_ROOT_DIR / "logs"
FIGURE_DIR = EXPERIMENT_DIR / "figures"
MAIN_SIMULATION_DIR = EXPERIMENT_DIR / "main_simulation_benchmark"
MODEL_SELECTION_BENCHMARK_DIR = EXPERIMENT_DIR / "model_selection_benchmark"
