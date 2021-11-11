#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module

from InnerEye.Common.metrics_constants import SEQUENCE_POSITION_HUE_NAME_PREFIX
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImagingFeatureType
from InnerEye.ML.reports.notebook_report import convert_to_html
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES
from InnerEye.ML.visualizers.model_hooks import HookBasedFeatureExtractor


def _tensor_as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class GradientBasedFeatureExtractor(HookBasedFeatureExtractor):
    """
    Base class for GradCam and BackPropagation classes.
    """

    def __init__(self, model: Module,
                 config: ScalarModelBase,
                 target_layer: Any,
                 target_pos: int = -1):
        if not config.is_classification_model:
            raise NotImplementedError("Visualizations maps with GradCam are only"
                                      "implemented for classification models.")
        super().__init__(model, target_layer)
        self.config = config
        self.hooks: List[Any] = []
        if config.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.encode_jointly = getattr(self.net, "encode_channels_jointly", True)
        self.imaging_feature_type = getattr(self.net, "imaging_feature_type", ImagingFeatureType.Image)
        self.num_non_image_features = getattr(self.net, "num_non_image_features", 0)
        self.target_label_index = target_pos
        self.logits = torch.Tensor()
        self.probabilities = torch.Tensor()

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_target_score(self) -> torch.Tensor:
        """
        Returns the target score i.e. logits for a positive predicted class,
        negative logits for negative target class.

        :param probabilities: Probabilities associated to the logits
        :param logits: Output of the network before Sigmoid
        """
        if self.logits.shape[-1] != 1:
            raise NotImplementedError("More than one output class")
        return torch.where(self.probabilities > 0.5, self.logits, -self.logits)

    def backward(self) -> None:
        """
        Defines the backward pass. Computes first the target scores to use to
        for backpropagation and then updates the `gradients` attribute based on
        the current value of the `logits` attribute (set in the forward pass).
        """
        target_scores = self.get_target_score()  # [B, num_targets, 1] or [B, 1]
        self.model.zero_grad()

        # If we have a sequence model, with potentially multiple labels.
        # Only backpropagate the gradients for the given target_pos
        if isinstance(self.config, SequenceModelBase):
            gradients_to_propagate = torch.zeros(target_scores.shape, device=self.device)
            gradients_to_propagate[:, self.target_label_index, :] = 1
        else:
            gradients_to_propagate = torch.ones(target_scores.shape, device=self.device)
        target_scores.backward(gradient=gradients_to_propagate)
        self.remove_hooks()


class GradCam(GradientBasedFeatureExtractor):
    """
    Class to generate GradCam maps for images, "Pseudo-GradCam" (i.e. ReLu(input x gradients))
    for non-images features of one batch for the given classification model. Tested and maintained for
    ImageEncoderWithMLP and RNNClassifier (models that take both images and non-imaging feautres as input).

    GradCam computes Relu(Gradients x Activations) at the output of the encoder of the network
    (before the global pooling layer). "PseudoGradCam" for non-imaging features denotes
    ReLu(input x gradients) for non-imaging features. "PseudoGradCam" is
    used to compare relative feature importance of various non-imaging features for the final classification
    task.
    """

    def __init__(self, model: Union[DeviceAwareModule, torch.nn.DataParallel],
                 config: ScalarModelBase) -> None:
        """

        :param model: The model to analyse
        :param config: The ScalarModelBase config defining the parameters of this model.
        """
        self.total_num_categorical_features = config.get_total_number_of_categorical_non_imaging_features()
        self.total_number_of_numerical_non_imaging_features = \
            config.get_total_number_of_numerical_non_imaging_features()
        self.is_non_imaging_model = config.is_non_imaging_model
        if self.is_non_imaging_model:
            super().__init__(model, config=config, target_layer=None)
        else:
            if isinstance(model, torch.nn.DataParallel):
                _model: DeviceAwareModule = model.module  # type: ignore
                target_layer = _model.get_last_encoder_layer_names()
                self.conv_in_3d = bool(_model.conv_in_3d)
            else:
                target_layer = model.get_last_encoder_layer_names()
                self.conv_in_3d = bool(model.conv_in_3d)
            super().__init__(model=model, config=config, target_layer=target_layer)
        self.gradients: Dict = {}
        self.activations: Dict = {}

    def backward_hook_fn(self, module: Module, grad_in: torch.Tensor, grad_out: torch.Tensor) -> None:
        """
        Backward hook to save the gradients per device (to allow GradCam to be computed
        with DataParallel models when training on multiple GPUs).
        """
        device = str(grad_out[0].get_device())
        if device not in self.gradients:
            self.gradients[device] = []
        self.gradients[device].append(grad_out[0].data.clone())

    def forward_hook_fn(self, module: Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Forward hook to save the activations of a given layer (per device to allow GradCam to be computed
        with DataParallel models when training on multiple GPUs.
        """
        device = str(output[0].get_device())
        if device not in self.activations:
            self.activations[device] = []
        if isinstance(output, tuple):
            self.activations[device].append([output[index].data.clone() for index in range(len(output))])
        else:
            self.activations[device].append(output.data.clone())

    def forward(self, *input) -> None:  # type: ignore
        """
        Triggers the call to the forward pass of the module. Prior to call the forward model function, we
        set the forward and backward passes. When calling this function, the `activations` attribute
        will containing the activations of the target layer for the given `input` batch passed as an
        argument to this function.
        """
        self.activations = {}
        if self.layer_name is not None:
            submodule = self.net
            for el in self.layer_name:
                submodule = submodule._modules[el]  # type: ignore
            target_layer = submodule
            self.hooks.append(target_layer.register_forward_hook(self.forward_hook_fn))
            self.hooks.append(target_layer.register_backward_hook(self.backward_hook_fn))  # type: ignore

        self.logits = self.model(*input)
        if isinstance(self.logits, List):
            self.logits = torch.nn.parallel.gather(self.logits, target_device=self.device)
        self.probabilities = torch.nn.Sigmoid()(self.logits)

    def backward(self) -> None:
        """
        Defines the backward pass. Computes first the target scores to use to
        for backpropagation and then updates the `gradients` attribute based on
        the current value of the `logits` attribute (set in the forward pass).
        """
        self.gradients = {}
        super().backward()

    def _get_image_grad_cam(self, input: List[torch.Tensor]) -> np.ndarray:
        """
        Get GradCam mps for images input. GradCam computes
        Relu(Gradients x Activations) at the output of the encoder of the network
        (before the global pooling layer).

        :param input: input batch
        :return: the GradCam maps
        """
        list_gradients = []
        list_activations = []
        # put all channels in one tensor per device
        for device in self.gradients:
            list_gradients.append(torch.stack(self.gradients[device], dim=1))  # [B, C_in, C_out, Z, X, Y]
            list_activations.append(torch.stack(self.activations[device], dim=1))  # [B, C_in, C_out, Z, X, Y]

        if self.config.use_gpu:
            activations = torch.nn.parallel.gather(list_activations, target_device=self.device)
            gradients = torch.nn.parallel.gather(list_gradients, target_device=self.device)

        else:
            assert len(list_activations) == 1
            activations = list_activations[0]
            gradients = list_gradients[0]
        self.gradients = {}
        self.activations = {}

        B, C_in = input[0].shape[:2]
        Z, X, Y = input[0].shape[-3:]
        B_act, _, C_act, Z_act, X_act, Y_act = activations.shape
        if self.conv_in_3d:
            weights = torch.mean(gradients, dim=(3, 4, 5), keepdim=True)
            Z_low = Z_act
        else:
            weights = torch.mean(gradients, dim=(4, 5), keepdim=True)
            Z_low = Z
        del list_gradients, gradients

        low_dim_cam = torch.nn.functional.relu(torch.mul(activations, weights).sum(dim=2))
        del weights, list_activations, activations

        # Case one separate encoding per channel i.e. one GradCam map per channel
        if not self.encode_jointly:
            if self.imaging_feature_type == ImagingFeatureType.Segmentation:
                assert low_dim_cam.shape == (B, C_in, Z_low, X_act, Y_act) \
                       or low_dim_cam.shape == (B, C_in / HDF5_NUM_SEGMENTATION_CLASSES, Z_low, X_act, Y_act)

            elif self.imaging_feature_type == ImagingFeatureType.Image:
                assert low_dim_cam.shape == (B, C_in, Z_low, X_act, Y_act)
        # Case one global encoding i.e. one GradCam map per image
        else:
            assert low_dim_cam.shape == (B, 1, Z_low, X_act, Y_act)

        grad_cam = torch.nn.functional.interpolate(
            low_dim_cam,
            (Z, X, Y),
            mode="trilinear"
        )
        return _tensor_as_numpy(grad_cam)

    def _get_non_imaging_grad_cam(self) -> np.ndarray:
        """
        Computes the "Pseudo GradCam" for non-imaging features i.e.
        ReLu(non_imaging_inputs x gradients).
        """
        assert self.non_image_input.grad is not None
        total_pseudo_cam_non_image = _tensor_as_numpy(torch.nn.functional.relu(
            torch.mul(self.non_image_input, self.non_image_input.grad)))
        batch_size = self.non_image_input.shape[0]
        non_image_input = _tensor_as_numpy(self.non_image_input)
        if self.total_num_categorical_features > 0:
            if len(total_pseudo_cam_non_image.shape) == 2:
                total_pseudo_cam_non_image = total_pseudo_cam_non_image.reshape(batch_size, 1, -1)
                non_image_input = non_image_input.reshape(batch_size, 1, -1)

            pseudo_cam_numerical = total_pseudo_cam_non_image[:, :,
                                   :self.total_number_of_numerical_non_imaging_features]

            pseudo_cam_one_hot = total_pseudo_cam_non_image[:, :,
                                 self.total_number_of_numerical_non_imaging_features:]
            categorical_input_one_hot = non_image_input[:, :, self.total_number_of_numerical_non_imaging_features:]

            # Back to "not one hot", only one value per feature is non zero
            batch_size, number_positions = pseudo_cam_one_hot.shape[:2]
            if isinstance(self.config, SequenceModelBase):
                pseudo_cam_categorical = np.zeros((batch_size, number_positions, len(self.config.categorical_columns)))
                for b in range(batch_size):
                    for t in range(number_positions):
                        # Some features come from sequence padding, for those the entire row is 0 i.e. the feature
                        # is not really one-hot encoded.
                        if np.any(categorical_input_one_hot[b, t] != 0):
                            pseudo_cam_categorical[b, t] = pseudo_cam_one_hot[
                                b, t, categorical_input_one_hot[b, t] != 0]
            else:
                # For a non-sequence model a categorical feature might appear several times for several channels but
                # there
                # is no padding. Hence we handle the conversion differently.
                pseudo_cam_categorical = pseudo_cam_one_hot[categorical_input_one_hot.cpu() != 0].reshape(
                    (batch_size, number_positions, -1))

            return np.concatenate([pseudo_cam_numerical, pseudo_cam_categorical], axis=2)
        else:
            return total_pseudo_cam_non_image

    def generate(self, input: List[torch.Tensor], target_position: int = -1, target_label_index: int = -1) \
            -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Generates the GradCam for images, PseudoGradCam for non-imaging features
        of one batch for the given classification model.

        GradCam computes Relu(Gradients x Activations) at the output of the encoder of the network
        (before the global pooling layer). "PseudoGradCam" for non-imaging features denotes
        ReLu(input x gradients) for non-imaging features. "PseudoGradCam" is used to compare relative feature importance
        of various non-imaging features for the final classification task.

        :param input: input image [B, C, Z, X, Y]
        :param target_position: in case of sequence model with multiple target weeks, specify which target
        position prediction should be visualized. By default the last one.
        :param target_label_index: index of the target label in the array of targets labels i.e. if target
        positions are [2,3,5], the target_label_index for position 3 is 1.
        :return: grad_cam: grad_cam maps [B, Z, X, Y]
        """
        self.target_label_index = target_label_index

        self.model.eval()
        if self.num_non_image_features > 0:
            self.non_image_input = input[1].clone().to(self.device).requires_grad_(True)
            self.forward(*[input[0], self.non_image_input])
        elif self.is_non_imaging_model:
            self.non_image_input = input[0].clone().to(self.device).requires_grad_(True)
            self.forward(self.non_image_input)
        else:
            self.forward(*input)
        self.backward()

        with torch.no_grad():
            grad_cam_image = None
            pseudo_cam_non_image = None
            if not self.is_non_imaging_model:
                grad_cam_image = self._get_image_grad_cam(input)
                if target_position > -1:
                    grad_cam_image = grad_cam_image[:, :(target_position + 1), ...]
            if self.num_non_image_features > 0 or self.is_non_imaging_model:
                pseudo_cam_non_image = self._get_non_imaging_grad_cam()
                if target_position > -1:
                    pseudo_cam_non_image = pseudo_cam_non_image[:, :(target_position + 1), ...]
        if self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            if self.encode_jointly:
                # In this case broadcasting can happen automatically in GuidedGradCam
                # computation.
                assert grad_cam_image is not None  # for mypy
                assert grad_cam_image.shape[1] == 1
            else:
                # Otherwise, copy GradCam output twice to compute GuidedGradCam (once for images and
                # once for segmentations).
                grad_cam_image = np.concatenate([grad_cam_image, grad_cam_image], axis=1)
        return grad_cam_image, pseudo_cam_non_image, _tensor_as_numpy(self.probabilities[:, target_label_index])


class GuidedBackPropagation(GradientBasedFeatureExtractor):
    """
    Class to compute GuidedBackPropagation maps for images features.
    """

    def __init__(self, model: Module, config: ScalarModelBase) -> None:
        super().__init__(model=model, config=config, target_layer=None)

    def guided_backprop_hook(self, module: Module, grad_in: torch.Tensor, grad_out: torch.Tensor) \
            -> Optional[Tuple[torch.Tensor]]:
        """
        Backward hook for guided Backpropagation.
        Propagate only positive gradient when backpropagating through ReLu layers.
        """
        # For all ReLU layers propagate only positive gradients
        if isinstance(module, torch.nn.ReLU):
            return torch.nn.functional.relu(grad_in[0]),
        return None

    def forward(self, *input):  # type: ignore
        """
        Triggers the call to the forward pass of the module and the registration of the backward hook.
        """
        for layer in self.net.modules():
            # Type check disabled: the type is correct but the PyTorch documentation is not.
            # noinspection PyTypeChecker
            self.hooks.append(layer.register_backward_hook(self.guided_backprop_hook))

        self.image_input_grad = input[0].clone().requires_grad_(True)
        if self.num_non_image_features > 0:
            self.logits = self.model(self.image_input_grad, input[1])
        else:
            self.logits = self.model(self.image_input_grad)
        if isinstance(self.logits, List):
            self.logits = torch.nn.parallel.gather(self.logits, target_device=self.device)
        self.probabilities = torch.nn.Sigmoid()(self.logits)

    def generate(self, input: List[torch.Tensor],
                 target_position: int = -1,
                 target_label_index: int = -1) -> np.ndarray:
        """
        Generate Guided Backpropagation maps for one input batch.

        :param input: input batch
        :param target_position: in case of sequence model with multiple target weeks, specify which target
        position prediction should be visualized. By default the last one.
        :param target_label_index: index of the target label in the array of targets labels i.e. if target
        positions are [2,3,5], the target_label_index for position 3 is 1.
        :return: guided backprop maps, size [B, C, Z, X, Y]
        """
        self.target_label_index = target_label_index
        self.model.eval()
        self.forward(*input)
        if self.config.use_gpu:
            torch.cuda.empty_cache()
        self.backward()

        B, C = input[0].shape[:2]
        Z, X, Y = input[0].shape[-3:]
        if self.imaging_feature_type == ImagingFeatureType.Segmentation:
            grads_of_one_hot = -_tensor_as_numpy(self.image_input_grad.grad)
            one_hot_input = _tensor_as_numpy(self.image_input_grad)
            backprop_map = grads_of_one_hot * one_hot_input
            backprop_map = backprop_map.reshape((B, -1, HDF5_NUM_SEGMENTATION_CLASSES, Z, X, Y))
            backprop_map = backprop_map.sum(axis=2)  # [B, C, Z, X, Y]
            if target_position > -1:
                backprop_map = backprop_map[:, :(target_position + 1), ...]
            return backprop_map
        elif self.imaging_feature_type == ImagingFeatureType.Image:
            backprop_map = self.image_input_grad.grad.detach().cpu().numpy().reshape((B, C, Z, X, Y))
            if target_position > -1:
                backprop_map = backprop_map[:, :(target_position + 1), ...]
            return backprop_map
        elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            grads = _tensor_as_numpy(self.image_input_grad.grad)  # [1, CLASSES, Z, X, Y] or [1, week, CLASSES, Z, X, Y]
            input_image = _tensor_as_numpy(self.image_input_grad)
            if len(grads.shape) == 5:
                grads = grads.reshape((B, -1, HDF5_NUM_SEGMENTATION_CLASSES + 1, Z, X, Y))
                input_image = input_image.reshape((B, -1, HDF5_NUM_SEGMENTATION_CLASSES + 1, Z, X, Y))
            one_hot_input = input_image[:, :, :-1, ...]
            grads_of_one_hot = grads[:, :, :-1, ...]
            backprop_map_segmentation = (grads_of_one_hot * one_hot_input).sum(axis=2)  # [B, C, Z, X, Y]
            backprop_map_image = grads[:, :, -1, ...]  # [B, C, Z, X, Y]
            if target_position > -1:
                backprop_map_segmentation = backprop_map_segmentation[:, :(target_position + 1), ...]
                backprop_map_image = backprop_map_image[:, :(target_position + 1), ...]
            return np.concatenate([backprop_map_segmentation, backprop_map_image], axis=1)  # [B, 2*C, Z, X, Y]
        else:
            raise ValueError("This imaging feature type is not supported.")


class VisualizationMaps:
    """
    Wrapper class to compute GradCam maps, GuidedGradCam and "Pseudo-GradCam" maps
    for a specific model.
    """

    def __init__(self, model: Union[DeviceAwareModule, torch.nn.DataParallel],
                 config: ScalarModelBase) -> None:
        self.config = config
        self.is_non_imaging_model = config.is_non_imaging_model
        self.grad_cam: GradCam = GradCam(model, config)
        if not self.is_non_imaging_model:
            self.guided_backprop: GuidedBackPropagation = GuidedBackPropagation(model, config)
            self.encode_channels_jointly: bool = self.guided_backprop.encode_jointly
            self.imaging_feature_type = self.grad_cam.imaging_feature_type

    def generate(self, input: List[torch.Tensor],
                 target_position: int = -1,
                 target_label_index: int = -1) \
            -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Generates the GuidedGradCam, GradCam and PseudoGradCam maps (for non-imaging data)
        for one batch of data.

        :param input: input batch sixe [B, C, Z, X, Y]
        :param target_position: in case of sequence model with multiple target weeks, specify which target
        position prediction should be visualized. By default the last one.
        :param target_label_index: index of the target label in the array of targets labels i.e. if target
        positions are [2,3,5], the target_label_index for position 3 is 1.
        :return: A tuple of GuidedGradCam maps (for image input), GradCam maps (for image input),
         PseudoGradCam (for non-image inputs), posteriors predicted for the given target position.
        by the model for this batch of data.
        """
        image_gcam, pseudo_cam_non_img, probability = self.grad_cam.generate(input, target_position, target_label_index)
        if self.is_non_imaging_model:
            image_guided_gcam = None
        else:
            guided_bp = self.guided_backprop.generate(input, target_position, target_label_index)
            image_guided_gcam = image_gcam * guided_bp
        return image_guided_gcam, \
               image_gcam, \
               pseudo_cam_non_img, \
               probability

    def save_visualizations_in_notebook(self,
                                        classification_sequence: Union[ScalarItem,
                                                                       List[ClassificationItemSequence[ScalarItem]]],
                                        input_batch: List[torch.Tensor],
                                        filenames: List[str],
                                        ground_truth_labels: np.ndarray,
                                        gradcam_dir: Path
                                        ) -> None:
        """
        Generate, plot and save the visualizations for one batch of
        data for a sequence model. The visualization are produced in
        a Jupyter Notebook for readability. There is one notebook generated
        for each subject. This notebook can be viewed in the AML UI. Additionally
        a HTML is produced containing only the cells' output.

        :param input_batch: input to the network
        :param classification_sequence: classification item for current batch
        :param filenames: a list of filenames for the plots size: [Batch]
        :param ground_truth_labels: the labels for this input_batch
        :param gradcam_dir: directory where to save the plots.
        """
        non_image_features = self.config.numerical_columns + self.config.categorical_columns
        has_non_image_features = len(non_image_features) > 0
        batch_size = len(filenames)
        if isinstance(self.config, SequenceModelBase):
            target_indices = self.config.get_target_indices()
            if target_indices is None:
                target_indices = [-1]
        else:
            target_indices = [-1]
        for label_index in range(len(target_indices)):
            target_position = target_indices[label_index]
            current_output_dir = self.config.visualization_folder / f"{SEQUENCE_POSITION_HUE_NAME_PREFIX}_" \
                                                                    f"{target_position}"
            current_output_dir.mkdir(exist_ok=True)
            guided_grad_cams, grad_cams, pseudo_cam_non_img, probas = self.generate(input_batch,
                                                                                    target_position,
                                                                                    label_index)
            for i in range(batch_size):
                if not self.is_non_imaging_model:
                    non_imaging_labels = self._get_non_imaging_plot_labels(
                        classification_sequence,  # type: ignore
                        non_image_features,
                        index=i,
                        target_position=target_position)
                    if isinstance(self.config, SequenceModelBase):
                        image = self._get_image_attributes_for_sequence_item(classification_sequence,  # type: ignore
                                                                             index=i,
                                                                             target_position=target_position)
                    else:
                        image = self._get_image_attributes_for_scalar_item(classification_sequence, i)  # type: ignore

                    # Need to temporarily save the variables to access them from the notebook.
                    # Because papermill does not support passing numpy array as parameters.
                    np.save(str(gradcam_dir / "image.npy"), image)
                    assert grad_cams is not None
                    np.save(str(gradcam_dir / "gradcam.npy"), grad_cams[i])
                    assert guided_grad_cams is not None
                    np.save(str(gradcam_dir / "guided_grad_cam.npy"), guided_grad_cams[i])
                    if has_non_image_features:
                        assert pseudo_cam_non_img is not None
                        np.save(str(gradcam_dir / "non_image_pseudo_cam.npy"), pseudo_cam_non_img[i])
                    has_image_features = True
                else:
                    non_imaging_labels = self._get_non_imaging_plot_labels(
                        classification_sequence, non_image_features, index=i, target_position=target_position)
                    has_non_image_features = True
                    has_image_features = False
                    self.encode_channels_jointly = False
                    self.imaging_feature_type = ImagingFeatureType.Image
                    assert pseudo_cam_non_img is not None
                    np.save(str(gradcam_dir / "non_image_pseudo_cam.npy"), pseudo_cam_non_img[i])

                current_label = ground_truth_labels[i, label_index]

                # If the label is NaN it means that we don't have data for this position and
                # we used padding for the input. Hence do not save visualizations for this position.
                if not np.isnan(current_label):
                    params_dict = dict(subject_id=filenames[i],
                                       target_position=target_position,
                                       gradcam_dir=str(gradcam_dir),
                                       has_non_image_features=has_non_image_features,
                                       probas=str(probas[i]),
                                       ground_truth_labels=str(current_label),
                                       non_image_labels=non_imaging_labels,
                                       encode_jointly=self.encode_channels_jointly,
                                       imaging_feature_type=self.imaging_feature_type.value,
                                       has_image_features=has_image_features,
                                       value_image_and_segmentation=ImagingFeatureType.ImageAndSegmentation.value, )

                    result_path = str(current_output_dir.joinpath(f"{filenames[i]}.ipynb"))
                    import papermill
                    papermill.execute_notebook(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                            "gradcam_visualization.ipynb"),
                                               result_path,
                                               parameters=params_dict,
                                               progress_bar=False)
                    convert_to_html(Path(result_path))

    def _get_non_imaging_plot_labels(self, classification_item: Union[ScalarItem,
                                                                      List[ClassificationItemSequence[ScalarItem]]],
                                     non_image_features: List[str],
                                     index: int,
                                     target_position: int = -1) -> List[str]:
        """
        Gets labels to use for the plots of non-imaging feature importance.

        :param classification_item: The classification item for which the return the
        label (can vary from subject to subject as they might not all contain the same
        position in case of a sequence model).
        :param non_image_features: The name of the imaging features used by the model.
        :param index: The index of the subject in the batch (used only for sequence models).
        :return: the labels (list of string)
        """
        if isinstance(self.config, SequenceModelBase):
            channels = []
            for item in classification_item[index].items:  # type: ignore
                if (item.metadata.sequence_position - self.config.min_sequence_position_value) <= target_position \
                        or target_position == -1:
                    channels.append(item.metadata.sequence_position)
            return [f"{col}_{channel}" for channel in channels for col in
                    non_image_features]  # type: ignore
        else:
            non_imaging_labels = []
            non_image_features = self.config.numerical_columns + self.config.categorical_columns
            non_image_feature_channels_dict = self.config.get_non_image_feature_channels_dict()
            for col in non_image_features:
                non_imaging_labels.extend(
                    [f"{col}_{channel}" for channel in non_image_feature_channels_dict[col]])  # type: ignore
            return non_imaging_labels

    def _get_image_attributes_for_sequence_item(self,
                                                classification_sequence: List[ClassificationItemSequence[
                                                    ScalarItem]],
                                                index: int,
                                                target_position: int) -> np.ndarray:
        """
        Extract the image and/or the segmentation for the classification item to be able to
        produce the visualizations.

        :param classification_sequence: The classification sequence for which to plot (contains
        the entire batch)
        :param index: the exact subject for which to plot.
        :return: An array containing the imaging input to plot.
        """
        images = []
        segmentations = []
        for item in classification_sequence[index].items:
            if (item.metadata.sequence_position - self.config.min_sequence_position_value) <= target_position \
                    or target_position == -1:
                if self.imaging_feature_type == ImagingFeatureType.Segmentation:
                    segmentations.append(item.segmentations)
                elif self.imaging_feature_type == ImagingFeatureType.Image:
                    images.append(item.images)
                elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
                    segmentations.append(item.segmentations)
                    images.append(item.images)
        if self.imaging_feature_type == ImagingFeatureType.Image:
            return _tensor_as_numpy(torch.cat(images, dim=0)).astype(float)  # type: ignore
        elif self.imaging_feature_type == ImagingFeatureType.Segmentation:
            return _tensor_as_numpy(torch.cat(segmentations, dim=0)).astype(int)  # type: ignore
        elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            return np.concatenate(
                [_tensor_as_numpy(torch.cat(images, dim=0)).astype(float),  # type: ignore
                 _tensor_as_numpy(torch.cat(segmentations, dim=0)).astype(int)],  # type: ignore
                axis=0)
        else:
            raise ValueError(f"imaging_feature_type not recognized: {self.imaging_feature_type}")

    def _get_image_attributes_for_scalar_item(self, classification_item: ScalarItem, index: int) -> np.ndarray:
        """
        Extract the image and/or the segmentation for the classification item to be able to
        produce the visualizations.

        :param classification_item: The classification items for which to plot (contains the
        entire batch data).
        :param index: the exact subject for which to plot.
        :return: An array containing the imaging input to plot.
        """
        if self.imaging_feature_type == ImagingFeatureType.Segmentation:
            if classification_item.segmentations is None:
                raise ValueError("Expected classification_item.segmentations to not be None")
            return _tensor_as_numpy(classification_item.segmentations[index]).astype(int)
        elif self.imaging_feature_type == ImagingFeatureType.Image:
            return _tensor_as_numpy(classification_item.images[index]).astype(float)
        elif self.imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            image = _tensor_as_numpy(classification_item.images[index]).astype(float)  # [C, Z, X, Y]
            assert classification_item.segmentations is not None  # for mypy
            seg = _tensor_as_numpy(classification_item.segmentations[index]).astype(int)
            return np.concatenate([seg, image], axis=0)
        else:
            raise ValueError("The provided imaging feature type is not supported.")
