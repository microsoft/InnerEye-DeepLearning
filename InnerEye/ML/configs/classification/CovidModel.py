import logging
import random
import math
from pathlib import Path

from typing import Any, Callable, List

import PIL
import numpy as np
import pandas as pd
import param
import torch

from PIL import Image
from torch.nn import ModuleList, ModuleDict
from pytorch_lightning import LightningModule
from torchvision.transforms import Compose

from InnerEye.Common.common_util import ModelProcessing, get_best_epoch_results_path
from InnerEye.Common.metrics_constants import LoggingColumns

from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName

from InnerEye.ML.SSL.lightning_modules.ssl_classifier_module import SSLClassifier
from InnerEye.ML.SSL.utils import create_ssl_encoder, create_ssl_image_classifier, load_yaml_augmentation_config
from InnerEye.ML.augmentations.transform_pipeline import create_cxr_transforms_from_config
from InnerEye.ML.common import ModelExecutionMode

from InnerEye.ML.configs.ssl.CXR_SSL_configs import path_linear_head_augmentation_cxr
from InnerEye.ML.deep_learning_config import LRSchedulerType, MultiprocessingStartMethod, \
    OptimizerType

from InnerEye.ML.lightning_metrics import Accuracy05
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImagingFeatureType
from InnerEye.ML.model_config_base import ModelTransformsPerExecutionMode
from InnerEye.ML.model_testing import MODEL_OUTPUT_CSV

from InnerEye.ML.configs.ssl.CovidContainers import COVID_DATASET_ID
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.metrics_dict import MetricsDict, DataframeLogger


class CovidModel(ScalarModelBase):
    """
    Model to train a CovidDataset model from scratch or finetune from SSL-pretrained model.

    For AML you need to provide the run_id of your SSL training job as a command line argument
    --pretraining_run_recovery_id=id_of_your_ssl_model, this will download the checkpoints of the run to your
    machine and load the corresponding pretrained model.

    To recover from a particular checkpoint from your SSL run e.g. "recovery_epoch=499.ckpt" please use the
    --name_of_checkpoint argument.
    """
    use_pretrained_model = param.Boolean(default=False, doc="If True, start training from a model pretrained with SSL."
                                                            "If False, start training a DenseNet model from scratch"
                                                            "(random initialization).")
    freeze_encoder = param.Boolean(default=False, doc="Whether to freeze the pretrained encoder or not.")
    name_of_checkpoint = param.String(default=None, doc="Filename of checkpoint to use for recovery")
    test_set_ids_csv = param.String(default=None,
                                    doc="Name of the csv file in the dataset folder with the test set ids. The dataset"
                                        "is expected to have a 'series' and a 'subject' column. The subject column"
                                        "is assumed to contain unique ids.")

    def __init__(self, covid_dataset_id: str = COVID_DATASET_ID, **kwargs: Any):
        super().__init__(loss_type=ScalarLoss.CustomClassification,
                         class_names=['CVX0', 'CVX1', 'CVX2', 'CVX3'],
                         max_num_gpus=1,
                         azure_dataset_id=covid_dataset_id,
                         subject_column="series",
                         image_file_column="filepath",
                         label_value_column="final_label",
                         non_image_feature_channels=[],
                         numerical_columns=[],
                         use_mixed_precision=False,
                         num_dataload_workers=12,
                         multiprocessing_start_method=MultiprocessingStartMethod.fork,
                         train_batch_size=64,
                         optimizer_type=OptimizerType.Adam,
                         num_epochs=50,
                         l_rate_scheduler=LRSchedulerType.Step,
                         l_rate_step_gamma=1.0,
                         l_rate_multi_step_milestones=None,
                         inference_on_val_set=True,
                         ensemble_inference_on_val_set=True,
                         should_validate=False)  # validate only after adding kwargs
        self.num_classes = 4
        self.add_and_validate(kwargs)

    def validate(self) -> None:
        self.l_rate = 1e-5 if self.use_pretrained_model else 1e-4
        super().validate()
        if not self.use_pretrained_model and self.freeze_encoder:
            raise ValueError("No encoder to freeze when training from scratch. You requested training from scratch and"
                             "encoder freezing.")

    def should_generate_multilabel_report(self) -> bool:
        return False

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        if self.test_set_ids_csv:
            test_set_ids_csv = self.local_dataset / self.test_set_ids_csv
            test_series = pd.read_csv(test_set_ids_csv).series

            all_series = dataset_df.series.values
            check_all_test_series = all(test_series.isin(all_series))
            if not check_all_test_series:
                raise ValueError(f"Not all test series from {test_set_ids_csv} were found in the dataset.")

            test_set_subjects = dataset_df[dataset_df.series.isin(test_series)].subject.values
            train_and_val_series = dataset_df[~dataset_df.subject.isin(test_set_subjects)].series.values
            random.seed(42)
            random.shuffle(train_and_val_series)
            num_val_samples = math.floor(len(train_and_val_series) / 9)
            val_series = train_and_val_series[:num_val_samples]
            train_series = train_and_val_series[num_val_samples:]

            logging.info(f"Dropped {len(all_series) - (len(test_series) + len(train_and_val_series))} series "
                         f"due to subject overlap with test set.")
            return DatasetSplits.from_subject_ids(dataset_df,
                                                  train_ids=train_series,
                                                  val_ids=val_series,
                                                  test_ids=test_series,
                                                  subject_column="series",
                                                  group_column="subject")
        else:
            return DatasetSplits.from_proportions(dataset_df,
                                                  proportion_train=0.8,
                                                  proportion_val=0.1,
                                                  proportion_test=0.1,
                                                  subject_column="series",
                                                  group_column="subject",
                                                  shuffle=True)

    # noinspection PyTypeChecker
    def get_image_transform(self) -> ModelTransformsPerExecutionMode:
        config = load_yaml_augmentation_config(path_linear_head_augmentation_cxr)
        train_transforms = Compose(
            [DicomPreparation(), create_cxr_transforms_from_config(config, apply_augmentations=True)])
        val_transforms = Compose(
            [DicomPreparation(), create_cxr_transforms_from_config(config, apply_augmentations=False)])

        return ModelTransformsPerExecutionMode(train=train_transforms,
                                               val=val_transforms,
                                               test=val_transforms)

    def create_model(self) -> LightningModule:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.use_pretrained_model:
            path_to_checkpoint = self._get_ssl_checkpoint_path()

            model = create_ssl_image_classifier(
                num_classes=self.num_classes,
                pl_checkpoint_path=str(path_to_checkpoint),
                freeze_encoder=self.freeze_encoder)

        else:
            encoder = create_ssl_encoder(encoder_name=EncoderName.densenet121.value)
            model = SSLClassifier(num_classes=self.num_classes,
                                  encoder=encoder,
                                  freeze_encoder=self.freeze_encoder,
                                  class_weights=None)
        # Next args are just here because we are using this model within an InnerEyeContainer
        model.imaging_feature_type = ImagingFeatureType.Image  # type: ignore
        model.num_non_image_features = 0  # type: ignore
        model.encode_channels_jointly = True  # type: ignore
        return model

    def _get_ssl_checkpoint_path(self) -> Path:
        # Get the SSL weights from the AML run provided via "pretraining_run_recovery_id" command line argument.
        # Accessible via pretraining_run_checkpoints field of the config.
        assert self.pretraining_run_checkpoints is not None
        assert isinstance(self.pretraining_run_checkpoints, RunRecovery)
        ssl_path = self.checkpoint_folder / "ssl_checkpoint.ckpt"

        if not ssl_path.exists():  # for test (when it is already present) we don't need to redo this.
            if self.name_of_checkpoint is not None:
                logging.info(f"Using checkpoint: {self.name_of_checkpoint} as starting point.")
                path_to_checkpoint = self.pretraining_run_checkpoints.checkpoints_roots[0] / self.name_of_checkpoint
            else:
                path_to_checkpoint = self.pretraining_run_checkpoints.get_best_checkpoint_paths()[0]
                if not path_to_checkpoint.exists():
                    logging.info("No best checkpoint found for this model. Getting the latest recovery "
                                 "checkpoint instead.")
                    path_to_checkpoint = self.pretraining_run_checkpoints.get_recovery_checkpoint_paths()[0]
            assert path_to_checkpoint.exists()
            path_to_checkpoint.rename(ssl_path)
        return ssl_path

    def pre_process_dataset_dataframe(self) -> None:
        pass

    @staticmethod
    def get_loss_function() -> Callable:
        import torch
        import torch.nn.functional as F

        def custom_loss(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            labels = torch.argmax(labels, dim=-1)
            return F.cross_entropy(input=output, target=labels, reduction="sum")

        return custom_loss

    def get_post_loss_logits_normalization_function(self) -> Callable:
        return torch.nn.Softmax()

    def create_metric_computers(self) -> ModuleDict:
        return ModuleDict({MetricsDict.DEFAULT_HUE_KEY: ModuleList([Accuracy05()])})

    def compute_and_log_metrics(self,
                                logits: torch.Tensor,
                                targets: torch.Tensor,
                                subject_ids: List[str],
                                is_training: bool,
                                metrics: ModuleDict,
                                logger: DataframeLogger,
                                current_epoch: int,
                                data_split: ModelExecutionMode) -> None:
        posteriors = self.get_post_loss_logits_normalization_function()(logits)
        labels = torch.argmax(targets, dim=-1)
        metric = metrics[MetricsDict.DEFAULT_HUE_KEY][0]
        metric(posteriors, labels)

        per_subject_outputs = zip(subject_ids, posteriors.tolist(), targets.tolist())
        for subject, model_output, target in per_subject_outputs:
            for i in range(len(self.target_names)):
                logger.add_record({
                    LoggingColumns.Epoch.value: current_epoch,
                    LoggingColumns.Patient.value: subject,
                    LoggingColumns.Hue.value: self.target_names[i],
                    LoggingColumns.ModelOutput.value: model_output[i],
                    LoggingColumns.Label.value: target[i],
                    LoggingColumns.DataSplit.value: data_split.value
                })

    def generate_custom_report(self, report_dir: Path, model_proc: ModelProcessing) -> Path:
        """
        Generate a custom report for the Covid model. This report will read the file model_output.csv generated for
        the training, validation or test sets and compute both the multiclass accuracy and the accuracy for each of the
        hierarchical tasks.
        :param report_dir: Directory report is to be written to
        :param model_proc: Whether this is a single or ensemble model (model_output.csv will be located in different
        paths for single vs ensemble runs.)
        """

        label_prefix = LoggingColumns.Label.value
        output_prefix = LoggingColumns.ModelOutput.value

        def get_output_csv_path(mode: ModelExecutionMode) -> Path:
            p = get_best_epoch_results_path(mode=mode, model_proc=model_proc)
            return self.outputs_folder / p / MODEL_OUTPUT_CSV

        def get_labels_and_predictions(df: pd.DataFrame) -> pd.DataFrame:
            """
            Given a dataframe with predictions for a single subject, returns the label and model output for the
            tasks: CVX03vs12, CVX0vs3, CVX1vs2 and multiclass.
            """
            labels = []
            predictions = []
            for target in self.target_names:
                target_df = df[df[LoggingColumns.Hue.value] == target]
                predictions.append(target_df[output_prefix].item())
                labels.append(target_df[label_prefix].item())

            pred_cvx03vs12 = predictions[1] + predictions[2]
            label_cvx03vs12 = 1 if labels[1] or labels[2] else 0

            if (predictions[0] + predictions[3]) != 0:
                pred_cvx0vs3 = predictions[3] / (predictions[0] + predictions[3])
            else:
                pred_cvx0vs3 = np.NaN
            label_cvx0vs3 = 0 if labels[0] else 1 if labels[3] else np.NaN

            if (predictions[1] + predictions[2]) != 0:
                pred_cvx1vs2 = predictions[2] / (predictions[1] + predictions[2])
            else:
                pred_cvx1vs2 = np.NaN
            label_cvx1vs2 = 0 if labels[1] else 1 if labels[2] else np.NaN

            return pd.DataFrame.from_dict({LoggingColumns.Patient.value: [df.iloc[0][LoggingColumns.Patient.value]],
                                           output_prefix: [np.argmax(predictions)],
                                           label_prefix: [np.argmax(labels)],
                                           f"{output_prefix}_CVX03vs12": pred_cvx03vs12,
                                           f"{label_prefix}_CVX03vs12": label_cvx03vs12,
                                           f"{output_prefix}_CVX0vs3": pred_cvx0vs3,
                                           f"{label_prefix}_CVX0vs3": label_cvx0vs3,
                                           f"{output_prefix}_CVX1vs2": pred_cvx1vs2,
                                           f"{label_prefix}_CVX1vs2": label_cvx1vs2})

        def get_per_task_output_and_labels(df: pd.DataFrame) -> pd.DataFrame:
            df = df.groupby(LoggingColumns.Patient.value, as_index=False).apply(get_labels_and_predictions).reset_index(drop=True)
            return df

        def get_report_section(df: pd.DataFrame, data_split: ModelExecutionMode) -> str:
            def compute_binary_accuracy(model_outputs: pd.Series, labels: pd.Series) -> float:
                non_nan_indices = model_outputs.notna()
                return ((model_outputs[non_nan_indices] > .5) == labels[non_nan_indices]).mean()

            outputs_and_labels = get_per_task_output_and_labels(df)
            cvx03vs12_indices = (outputs_and_labels[f"{label_prefix}_CVX03vs12"] == 1)
            cvx03vs12_accuracy = compute_binary_accuracy(model_outputs=outputs_and_labels[f"{output_prefix}_CVX03vs12"],
                                                         labels=outputs_and_labels[f"{label_prefix}_CVX03vs12"])
            cvx0vs3_outputs_and_labels = outputs_and_labels[~cvx03vs12_indices]
            cvx0vs3_accuracy = compute_binary_accuracy(model_outputs=cvx0vs3_outputs_and_labels[f"{output_prefix}_CVX0vs3"],
                                                       labels=cvx0vs3_outputs_and_labels[f"{label_prefix}_CVX0vs3"])
            cvx1vs2_outputs_and_labels = outputs_and_labels[cvx03vs12_indices]
            cvx1vs2_accuracy = compute_binary_accuracy(model_outputs=cvx1vs2_outputs_and_labels[f"{output_prefix}_CVX1vs2"],
                                                       labels=cvx1vs2_outputs_and_labels[f"{label_prefix}_CVX1vs2"])
            multiclass_acc = (outputs_and_labels[output_prefix] == outputs_and_labels[label_prefix]).mean()  # type: ignore

            report_section_text = f"{data_split.value}\n"
            report_section_text += f"CVX03vs12 Accuracy: {cvx03vs12_accuracy:.4f}\n"

            report_section_text += f"CVX0vs3 Accuracy: {cvx0vs3_accuracy:.4f}\n"
            nan_in_cvx0vs3 = cvx0vs3_outputs_and_labels[f"{output_prefix}_CVX0vs3"].isna().sum()
            if nan_in_cvx0vs3 > 0:
                report_section_text += f"Warning: CVX0vs3 accuracy was computed skipping {nan_in_cvx0vs3} NaN model outputs.\n"

            report_section_text += f"CVX1vs2 Accuracy: {cvx1vs2_accuracy:.4f}\n"
            nan_in_cvx1vs2 = cvx1vs2_outputs_and_labels[f"{output_prefix}_CVX1vs2"].isna().sum()
            if nan_in_cvx1vs2 > 0:
                report_section_text += f"Warning: CVX1vs2 accuracy was computed skipping {nan_in_cvx1vs2} NaN model outputs.\n"

            report_section_text += f"Multiclass Accuracy: {multiclass_acc:.4f}\n"
            report_section_text += "\n"

            return report_section_text

        train_csv_path = get_output_csv_path(ModelExecutionMode.TRAIN)
        val_csv_path = get_output_csv_path(ModelExecutionMode.VAL)
        test_csv_path = get_output_csv_path(ModelExecutionMode.TEST)

        report_text = ""

        if train_csv_path.exists():
            train_df = pd.read_csv(train_csv_path)
            report_text += get_report_section(train_df, ModelExecutionMode.TRAIN)

        if val_csv_path.exists():
            val_df = pd.read_csv(val_csv_path)
            report_text += get_report_section(val_df, ModelExecutionMode.VAL)

        if test_csv_path.exists():
            test_df = pd.read_csv(test_csv_path)
            report_text += get_report_section(test_df, ModelExecutionMode.TEST)

        report = report_dir / "report.txt"
        report.write_text(report_text)

        logging.info(report_text)

        return report


class DicomPreparation:
    def __call__(self, item: torch.Tensor) -> PIL.Image:
        # Item will be of dimension [C, Z, X, Y]
        images = item.numpy()
        assert images.shape[0] == 1 and images.shape[1] == 1
        images = images.reshape(images.shape[2:])
        normalized_image = (images - images.min()) * 255. / (images.max() - images.min())
        image = Image.fromarray(normalized_image).convert("L")
        return image
