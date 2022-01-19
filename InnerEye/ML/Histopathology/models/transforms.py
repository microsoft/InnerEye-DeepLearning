#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Mapping, Sequence, Union

import PIL.Image
import torch
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform
from torchvision.transforms.functional import to_tensor

from InnerEye.ML.Histopathology.models.encoders import TileEncoder

PathOrString = Union[Path, str]


def load_pil_image(image_path: PathOrString) -> PIL.Image.Image:
    """Load a PIL image in RGB format from the given path"""
    return PIL.Image.open(image_path).convert('RGB')


def load_image_as_tensor(image_path: PathOrString) -> torch.Tensor:
    """Load an image as a tensor from the given path"""
    pil_image = load_pil_image(image_path)
    return to_tensor(pil_image)


def load_image_stack_as_tensor(image_paths: Sequence[PathOrString],
                               progress: bool = False) -> torch.Tensor:
    """Load a batch of images of the same size as a tensor from the given paths"""
    loading_generator = (load_image_as_tensor(path) for path in image_paths)
    if progress:
        from tqdm import tqdm
        loading_generator = tqdm(loading_generator, desc="Loading image stack",
                                 total=len(image_paths), leave=False)
    image_tensors = list(loading_generator)
    return torch.stack(image_tensors, dim=0)


class LoadTiled(MapTransform):
    """Dictionary transform to load an individual image tile as a tensor from an input path"""

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        :param keys: Key(s) for the image path(s) in the input dictionary.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = load_image_as_tensor(data[key])
        return out_data


class LoadTilesBatchd(MapTransform):
    """Dictionary transform to load a batch of image tiles as a tensor from a list of input paths"""

    # Cannot reuse MONAI readers because they support stacking only images with no channels
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False,
                 progress: bool = False) -> None:
        """
        :param keys: Key(s) for the image path(s) in the input dictionary.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        :param progress: Whether to display a tqdm progress bar.
        """
        super().__init__(keys, allow_missing_keys)
        self.progress = progress

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = load_image_stack_as_tensor(data[key], progress=self.progress)
        return out_data


class EncodeTilesBatchd(MapTransform):
    """Dictionary transform to extract features from a batch tensor of image tiles"""

    def __init__(self,
                 keys: KeysCollection,
                 encoder: TileEncoder,
                 allow_missing_keys: bool = False,
                 chunk_size: int = 0) -> None:
        """
        :param keys: Key(s) for the image path(s) in the input dictionary.
        :param encoder: The tile encoder to use for feature extraction.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        :param chunk_size: if > 0, extracts features in chunks of size chunk_size.
        """
        super().__init__(keys, allow_missing_keys)
        self.encoder = encoder
        self.chunk_size = chunk_size

    @torch.no_grad()
    def _encode_tiles(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        if self.chunk_size > 0:
            embeddings = []
            chunks = torch.split(images, self.chunk_size)
            # TODO parallelize encoding - keep metadata and images aligned
            for chunk in chunks:
                chunk_embeddings = self._encode_images(chunk, device)
                embeddings.append(chunk_embeddings)
            return torch.cat(embeddings)
        else:
            return self._encode_images(images, device)

    def _encode_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        embeddings = self.encoder(images)
        del images
        torch.cuda.empty_cache()
        return embeddings

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = self._encode_tiles(data[key])
        return out_data
