#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from InnerEyeDataQuality.datasets.cifar10_utils import get_cifar10_label_names
from InnerEyeDataQuality.deep_learning.metrics.sample_metrics import SampleMetrics
from InnerEyeDataQuality.deep_learning.metrics.plots_tensorboard import (get_scatter_plot, plot_disagreement_per_sample,
                                                                 plot_excluded_cases_coteaching)
from InnerEyeDataQuality.deep_learning.transforms import ToNumpy
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


@dataclass()
class JointMetrics():
    """
    Stores metrics for co-teaching models.
    """
    num_samples: int
    num_epochs: int
    dataset: Optional[Any] = None
    ambiguous_mislabelled_ids: np.ndarray = None
    clear_mislabelled_ids: np.ndarray = None
    true_label_entropy: np.ndarray = None
    plot_dropped_images: bool = False

    def reset(self) -> None:
        self.kl_divergence_symmetric = np.full([self.num_samples], np.nan)
        self.active = False

    def __post_init__(self) -> None:
        self.reset()
        self.prediction_disagreement = np.zeros([self.num_samples, self.num_epochs], dtype=np.bool)
        self._initialise_dataset_properties()
        self.case_drop_histogram = np.zeros([self.num_samples, self.num_epochs + 1], dtype=np.bool)
        if not isinstance(self.clear_mislabelled_ids, np.ndarray) or not isinstance(self.ambiguous_mislabelled_ids,
                                                                                    np.ndarray):
            return

        self.true_mislabelled_ids = np.concatenate([self.ambiguous_mislabelled_ids, self.clear_mislabelled_ids], axis=0)
        self.case_drop_histogram[self.clear_mislabelled_ids, -1] = True
        self.case_drop_histogram[self.ambiguous_mislabelled_ids, -1] = True

    def _initialise_dataset_properties(self) -> None:
        if self.dataset is not None:
            self.label_names = get_cifar10_label_names() if isinstance(self.dataset, CIFAR10) \
                else self.dataset.get_label_names()
            self.dataset.transform = ToNumpy()  # type: ignore

    def log_results(self, writer: SummaryWriter, epoch: int, sample_metrics: SampleMetrics) -> None:
        if (not self.active) or (self.ambiguous_mislabelled_ids is None) or (self.clear_mislabelled_ids is None):
            return

        # KL Divergence between the two posteriors
        writer.add_scalars(main_tag='symmetric-kl-divergence', tag_scalar_dict={
            'all': np.nanmean(self.kl_divergence_symmetric),
            'ambiguous': np.nanmean(self.kl_divergence_symmetric[self.ambiguous_mislabelled_ids]),
            'clear_noise': np.nanmean(self.kl_divergence_symmetric[self.clear_mislabelled_ids])},
                           global_step=epoch)

        # Disagreement rate between the models
        writer.add_scalars(main_tag='disagreement_rate', tag_scalar_dict={
            'all': np.nanmean(self.prediction_disagreement[:, epoch]),
            'ambiguous': np.nanmean(self.prediction_disagreement[self.ambiguous_mislabelled_ids, epoch]),
            'clear_noise': np.nanmean(self.prediction_disagreement[self.clear_mislabelled_ids, epoch])},
                           global_step=epoch)

        # Add histogram for the loss values
        self.log_loss_values(writer, sample_metrics.loss_per_sample[:, epoch], epoch)

        # Add disagreement metrics
        fig = get_scatter_plot(self.true_label_entropy, self.kl_divergence_symmetric,
                               x_label="Label entropy", y_label="Symmetric-KL", y_lim=[0.0, 2.0])
        writer.add_figure('Sym-KL vs Label Entropy', figure=fig, global_step=epoch, close=True)

        fig = plot_disagreement_per_sample(self.prediction_disagreement, self.true_label_entropy)
        writer.add_figure('Disagreement of prediction', figure=fig, global_step=epoch, close=True)

        # Excluded cases diagnostics
        self.log_dropped_cases_metrics(writer=writer, epoch=epoch)

        # Every 10 epochs, display the dropped cases in the co-teaching algorithm
        if epoch % 10 and self.plot_dropped_images:
            self.log_dropped_images(writer=writer, predictions=sample_metrics.predictions, epoch=epoch)

        # Close all figures
        plt.close('all')

    def log_dropped_cases_metrics(self, writer: SummaryWriter, epoch: int) -> None:
        """
        Creates all diagnostics for dropped cases analysis.
        """
        entropy_sorted_indices = np.argsort(self.true_label_entropy)
        drop_cur_epoch_mask = self.case_drop_histogram[:, epoch]
        drop_cur_epoch_ids = np.where(drop_cur_epoch_mask)[0]
        is_sample_dropped = np.any(drop_cur_epoch_mask)
        title = None
        if is_sample_dropped:
            n_dropped = float(drop_cur_epoch_ids.size)
            average_label_entropy_dropped_cases = np.mean(self.true_label_entropy[drop_cur_epoch_mask])
            n_detected_mislabelled = np.intersect1d(drop_cur_epoch_ids, self.true_mislabelled_ids).size
            n_clean_dropped = int(n_dropped - n_detected_mislabelled)
            n_detected_mislabelled_ambiguous = np.intersect1d(drop_cur_epoch_ids, self.ambiguous_mislabelled_ids).size
            n_detected_mislabelled_clear = np.intersect1d(drop_cur_epoch_ids, self.clear_mislabelled_ids).size
            perc_detected_mislabelled = n_detected_mislabelled / n_dropped * 100
            perc_detected_clear_mislabelled = n_detected_mislabelled_clear / n_dropped * 100
            perc_detected_ambiguous_mislabelled = n_detected_mislabelled_ambiguous / n_dropped * 100
            title = f"Dropped Cases: Avg label entropy {average_label_entropy_dropped_cases:.3f}\n " \
                    f"Dropped cases: {n_detected_mislabelled} mislabelled ({perc_detected_mislabelled:.1f}%) - " \
                    f"{n_clean_dropped} clean ({(100 - perc_detected_mislabelled):.1f}%)\n" \
                    f"Num ambiguous mislabelled among detected cases: {n_detected_mislabelled_ambiguous}" \
                    f" ({perc_detected_ambiguous_mislabelled:.1f}%)\n" \
                    f"Num clear mislabelled among detected cases: {n_detected_mislabelled_clear}" \
                    f" ({perc_detected_clear_mislabelled:.1f}%)"
            writer.add_scalars(main_tag='Number of dropped cases', tag_scalar_dict={
                'clean_cases': n_clean_dropped,
                'all_mislabelled_cases': n_detected_mislabelled,
                'mislabelled_clear_cases': n_detected_mislabelled_clear,
                'mislabelled_ambiguous_cases': n_detected_mislabelled_ambiguous}, global_step=epoch)
            writer.add_scalar(tag="Percentage of mislabelled among dropped cases",
                              scalar_value=perc_detected_mislabelled, global_step=epoch)
        fig = plot_excluded_cases_coteaching(case_drop_mask=self.case_drop_histogram,
                                             entropy_sorted_indices=entropy_sorted_indices, title=title,
                                             num_epochs=self.num_epochs, num_samples=self.num_samples)
        writer.add_figure('Histogram of excluded cases', figure=fig, global_step=epoch, close=True)

    def log_loss_values(self, writer: SummaryWriter, loss_values: np.ndarray, epoch: int) -> None:
        """
        Logs histogram of loss values of one of the co-teaching models.
        """
        writer.add_histogram('loss/all', loss_values, epoch)
        writer.add_histogram('loss/ambiguous_noise', loss_values[self.ambiguous_mislabelled_ids], epoch)
        writer.add_histogram('loss/clear_noise', loss_values[self.clear_mislabelled_ids], epoch)

    def log_dropped_images(self, writer: SummaryWriter, predictions: np.ndarray, epoch: int) -> None:
        """
        Logs images dropped during co-teaching training
        """
        dropped_cases = np.where(self.case_drop_histogram[:, epoch])[0]
        if dropped_cases.size > 0 and self.dataset is not None:
            dropped_cases = dropped_cases[np.argsort(self.true_label_entropy[dropped_cases])]
            fig = self.plot_batch_images_and_labels(predictions, list_indices=dropped_cases[:64])
            writer.add_figure("Dropped images with lowest entropy", figure=fig, global_step=epoch, close=True)
            fig = self.plot_batch_images_and_labels(predictions, list_indices=dropped_cases[-64:])
            writer.add_figure("Dropped images with highest entropy", figure=fig, global_step=epoch, close=True)

            kept_cases = np.where(~self.case_drop_histogram[:, epoch])[0]
            kept_cases = kept_cases[np.argsort(self.true_label_entropy[kept_cases])]
            fig = self.plot_batch_images_and_labels(predictions, kept_cases[-64:])
            writer.add_figure("Kept images with highest entropy", figure=fig, global_step=epoch, close=True)

    def plot_batch_images_and_labels(self, predictions: np.ndarray, list_indices: np.ndarray) -> plt.Figure:
        """
        Plots of batch of images along with their labels and predictions. Noise cases are colored in red, clean cases
        in green. Images are assumed to be numpy images (use ToNumpy() transform).
        """
        assert self.dataset is not None
        fig, ax = plt.subplots(8, 8, figsize=(8, 10))
        ax = ax.ravel()
        for i, index in enumerate(list_indices):
            predicted = int(predictions[index])
            color = "red" if index in self.true_mislabelled_ids else "green"
            _, img, training_label = self.dataset.__getitem__(index)
            ax[i].imshow(img)
            ax[i].set_axis_off()
            ax[i].set_title(f"Label: {self.label_names[training_label]}\nPred: {self.label_names[predicted]}",
                            color=color, fontsize="x-small")
        return fig

