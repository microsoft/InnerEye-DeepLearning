#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from typing import Callable, Optional, Tuple, List, Generator

import numpy as np
import PIL.Image
import torch
import torchvision
from InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from InnerEyeDataQuality.datasets.tools import get_instance_noise_model
from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot
from pl_bolts.models.self_supervised.resnets import resnet50_bn
from InnerEyeDataQuality.datasets.cifar10h import CIFAR10H

def chunks(lst: List, n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class CIFAR10IDN(torchvision.datasets.CIFAR10):
    """
    Dataset class for the CIFAR10 dataset where target labels are sampled from a confusion matrix.
    """

    def __init__(self,
                 root: str,
                 train: bool,
                 noise_rate: float,
                 transform: Optional[Callable] = None,
                 download: bool = True,
                 use_fixed_labels: bool = True,
                 seed: int = 1
                 ) -> None:
        """
        :param root: The directory in which the CIFAR10 images will be stored.
        :param train: If True, creates dataset from training set, otherwise creates from test set.
        :param transform: Transform to apply to the images.
        :param download: Whether to download the dataset if it is not already in the local disk.
        :param noise_rate: Expected noise rate in the sampled labels.
        :param use_fixed_labels: If true labels are sampled only once and are kept fixed. If false labels are sampled at
                                 each get_item() function call from label distribution.
        :param seed: The random seed that defines which samples are train/test and which labels are sampled.
        """
        super().__init__(root, train=train, transform=transform, target_transform=None, download=download)
        self.seed = seed
        self.targets = np.array(self.targets, dtype=np.int64)  # type: ignore
        self.num_classes = np.unique(self.targets, return_counts=False).size
        self.num_samples = len(self.data)
        self.clean_targets = np.copy(self.targets)
        self.np_random_state = np.random.RandomState(seed)
        self.use_fixed_labels = use_fixed_labels
        self.indices = np.array(range(self.num_samples))
        logging.info(f"Preparing dataset: CIFAR10-IDN (N={self.num_samples})")

        # Set seed for torch operations
        initial_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Collect image embeddings
        embeddings = self.get_cnn_image_embeddings(self.data)
        targets = torch.from_numpy(self.targets)

        # Create label distribution for simulation of label adjudication
        label_counts = convert_labels_to_one_hot(self.clean_targets, n_classes=self.num_classes) if train else \
                       CIFAR10H.download_cifar10h_labels(self.root)
        self.label_distribution = LabelDistribution(seed, label_counts, temperature=1.0)

        # Add asymmetric noise on the labels
        self.noise_models = get_instance_noise_model(n=noise_rate,
                                                     dataset=zip(embeddings, targets),
                                                     labels=targets,
                                                     num_classes=self.num_classes,
                                                     feature_size=embeddings.shape[1],
                                                     norm_std=0.01,
                                                     seed=self.seed)

        if self.use_fixed_labels:
            # Sample target labels 
            self.targets = self.sample_labels_from_model()

            # Check the class distribution after sampling
            class_distribution = self.get_class_frequencies(targets=self.targets, num_classes=self.num_classes)
            noise_rate = np.mean(self.clean_targets != self.targets) * 100.0

            # Log dataset details
            logging.info(f"Class distribution (%) (true labels): {class_distribution * 100.0}")
            logging.info(f"Label noise rate: {noise_rate}")
        else: 
            self.targets = None

        # Restore initial state
        torch.set_rng_state(initial_state)

    def sample_labels_from_model(self, sample_index: Optional[int] = None) -> np.ndarray:
        """
        Samples class labels for each data point based on true label and instance dependent noise model
        """
        classes = [i for i in range(self.num_classes)]
        if sample_index is not None:
            _t = self.np_random_state.choice(classes, p=self.noise_models[sample_index]) 
        else:
            _t = [self.np_random_state.choice(classes, p=self.noise_models[i]) for i in range(self.num_samples)]
        return np.array(_t)

    @staticmethod
    def get_class_frequencies(targets: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Returns normalised frequency of each semantic class
        """
        assert targets.ndim == 1
        class_ids, class_counts = np.unique(targets, return_counts=True)
        class_distribution = np.zeros(num_classes, dtype=float)
        for ii in range(num_classes):
            if np.any(class_ids == ii):
                class_distribution[ii] = class_counts[class_ids == ii]/targets.size

        return class_distribution

    @staticmethod
    def get_cnn_image_embeddings(data: np.ndarray) -> torch.Tensor:
        """
        Extracts image embeddings using a pre-trained model
        """
        num_samples = data.shape[0]
        embeddings = list()
        data = torch.from_numpy(data).float().cuda()
        encoder = resnet50_bn(return_all_feature_maps=False, pretrained=True).cuda().eval()
        with torch.no_grad():
            for i in chunks(list(range(num_samples)), n=100):
                input = data[i].permute(0, 3, 1, 2)
                embeddings.append(encoder(input)[-1].cpu())
        return torch.cat(embeddings, dim=0).view(num_samples, -1)

    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        """
        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        img = PIL.Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.use_fixed_labels:
            target = self.targets[index]
        else:
            target = self.sample_labels_from_model(sample_index=index)

        return img, int(target)
