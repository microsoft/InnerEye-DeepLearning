#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.Common.common_util import is_gpu_tensor
from InnerEye.ML.utils.transforms import Compose3D, Transform3D
from Tests.ML.util import no_gpu_available


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
@pytest.mark.skipif(no_gpu_available, reason="Testing Transforms with GPU tesors requires a GPU")
def test_transform_compose_gpu() -> None:
    test_transform_compose(use_gpu=True)


def test_transform_compose(use_gpu: bool = False) -> None:
    class Identity(Transform3D[torch.Tensor]):
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:
            return self.get_gpu_tensor_if_possible(sample)

    class Square(Transform3D[torch.Tensor]):
        def __call__(self, sample: torch.Tensor) -> torch.Tensor:
            return self.get_gpu_tensor_if_possible(sample) ** 2

    a = torch.randint(low=2, high=4, size=[1])
    if use_gpu:
        a = a.cuda()

    # test that composition of multiple identity operations holds
    identity_compose = Compose3D([Identity(use_gpu=use_gpu)] * 3)
    a_t = identity_compose(a)
    assert torch.equal(Compose3D.apply(identity_compose, a), a_t)
    assert torch.equal(a_t, a)
    assert is_gpu_tensor(a_t) == use_gpu

    # test that composition of multiple square operations holds
    square_compose = Compose3D([Square(use_gpu=use_gpu)] * 3)
    assert torch.equal(square_compose(a), a ** 8)
    assert torch.equal(Compose3D.apply(square_compose, a), a ** 8)
