#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import torch
import torch.optim as optim

from InnerEye.ML.models.losses.cross_entropy import CrossEntropyLoss
# Set random seed
from InnerEye.ML.utils.supervised_criterion import BinaryCrossEntropyWithLogitsLoss, SupervisedLearningCriterion

torch.random.manual_seed(1)


def test_get_class_weights() -> None:
    target = torch.tensor([[2, 2, 1, 2, 2], [3, 3, 3, 3, 3]], dtype=torch.long)
    weights = CrossEntropyLoss._get_class_weights(target_labels=target, num_classes=4)
    assert torch.eq(weights, torch.tensor([0.00, 1.00, 0.25, 0.20])).all()


def test_cross_entropy_loss_forward_zero_loss() -> None:
    target = torch.tensor([[[0, 0, 0], [1, 1, 1]]], dtype=torch.float32)
    logits = torch.tensor([[[-1e9, -1e9, -1e9], [0, 0, 0]]], dtype=torch.float32)

    # Extract class indices
    loss_fn = CrossEntropyLoss(class_weight_power=0.0)
    loss = loss_fn(logits, target)

    assert torch.isclose(loss, torch.tensor([0.000]))


def test_cross_entropy_loss_forward_balanced() -> None:
    # target: one-hot, B=1, C=2, N=3 voxels. First two voxels are class 1, last is class 0.
    target = torch.tensor([[[0, 0, 1], [1, 1, 0]]], dtype=torch.float32)
    # logits: predicting class 1 (correctly) at first two voxels, 50-50 at last voxel.
    logits = torch.tensor([[[-1e9, -1e9, 0], [0, 0, 0]]], dtype=torch.float32)

    # Compute loss values for unbalanced case.
    loss_fn = CrossEntropyLoss(class_weight_power=0.0)
    loss = loss_fn(logits, target)
    # Loss is (nearly) all from last voxel: -log(0.5). This is averaged over all 3 voxels (divide by 3).
    expected = -1 * torch.log(torch.tensor(0.5)) / 3.0
    assert (torch.isclose(loss, expected))

    # Compute loss values for balanced case.
    loss_fn = CrossEntropyLoss(class_weight_power=1.0)
    loss = loss_fn(logits, target)
    # Class weights should be 4/3 for class 0, 3/3 for class 1 (inverses of class frequencies, normalized
    # to average to 1). Loss comes from the uncertainty on the last voxel which is class 0...
    expected = expected * 4 / 3
    assert (torch.isclose(loss, expected))


@pytest.mark.parametrize("is_segmentation", [True, False])
def test_cross_entropy_loss_forward_smoothing(is_segmentation: bool) -> None:
    target = torch.tensor([[[0, 0, 1], [1, 1, 0]]], dtype=torch.float32)
    smoothed_target = torch.tensor([[[0.1, 0.1, 0.9], [0.9, 0.9, 0.1]]], dtype=torch.float32)
    logits = torch.tensor([[[-10, -10, 0], [0, 0, 0]]], dtype=torch.float32)

    barely_smoothed_loss_fn: SupervisedLearningCriterion = BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0)
    smoothed_loss_fn: SupervisedLearningCriterion = BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0.1)
    if is_segmentation:
        # The two loss values are only expected to be the same when no class weighting takes place,
        # because weighting is done on the *unsmoothed* target values.
        # We can't use a completely unsmoothed loss function because it won't like non-one-hot targets.
        barely_smoothed_loss_fn = CrossEntropyLoss(class_weight_power=0.0, smoothing_eps=1e-9)
        smoothed_loss_fn = CrossEntropyLoss(class_weight_power=0.0, smoothing_eps=0.1)

    loss1 = barely_smoothed_loss_fn(logits, smoothed_target)
    loss2 = smoothed_loss_fn(logits, target)
    assert torch.isclose(loss1, loss2)


@pytest.mark.parametrize(["focal_loss_gamma", "loss_upper_bound", "class_weight_power"],
                         [(None, 1e-4, 1.0),
                          (2.0, 1e-7, 0.0),
                          (2.0, 1e-7, 1.0)])
def test_cross_entropy_loss_integration(focal_loss_gamma: float,
                                        loss_upper_bound: float,
                                        class_weight_power: float) -> None:
    """
    Solves a simple linear classification problem by training a multi-layer perceptron.
    Here the training objectives (cross-entropy and focal loss) are tested to see they function
    properly when they are optimised with a stochastic optimiser.
    """
    # Set a seed
    torch.random.manual_seed(1)

    # Define hyperparameters
    n_samples = 1000
    batch_size = 16
    n_epochs = 40

    # Set the input data (labels 1000x2, features 1000x50, 1000 samples, 50 dimensional features)
    features = torch.cat([torch.randn(n_samples // 2, 50),
                          torch.randn(n_samples // 2, 50) + 1.5], dim=0)
    indices = torch.cat([torch.zeros(n_samples // 2, dtype=torch.long),
                         torch.ones(n_samples // 2, dtype=torch.long)], dim=0)
    labels = torch.nn.functional.one_hot(indices, num_classes=2).float()

    # Shuffle the dataset
    perm = torch.randperm(n_samples)
    features = features[perm, :]
    labels = labels[perm, :]

    # Define a basic model (We actually don't even a non-linear unit to solve it)
    net = ToyNet()
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = CrossEntropyLoss(class_weight_power=class_weight_power, focal_loss_gamma=focal_loss_gamma)

    # Perform forward and backward passes
    net.train()
    epoch_losses = []
    loss = torch.empty(0)  # to ensure never unset
    for epoch_id in range(0, n_epochs):
        for beg_i in range(0, features.size(0), batch_size):
            x_batch = features[beg_i:beg_i + batch_size, :]
            y_batch = labels[beg_i:beg_i + batch_size, :]

            opt.zero_grad()
            # (1) Forward
            y_hat = net(x_batch)
            # (2) Compute diff
            loss = loss_fn(y_hat, y_batch)
            # (3) Compute gradients
            loss.backward()
            # (4) update weights
            opt.step()

        # Add final epoch loss to the list
        epoch_losses.append(loss.data.numpy())

    # And see if loss is decaying for a given problem
    assert epoch_losses[0] > 0.10
    assert epoch_losses[10] < epoch_losses[0]
    assert epoch_losses[15] < epoch_losses[5]
    assert epoch_losses[n_epochs - 1] < loss_upper_bound


def test_weighted_binary_cross_entropy_loss_forward_smoothing() -> None:
    target = torch.tensor([[1], [1], [1], [1], [1], [0]], dtype=torch.float32)
    smoothed_target = torch.tensor([[0.9], [0.9], [0.9], [0.9], [0.9], [0.1]], dtype=torch.float32)
    logits = torch.tensor([[-10], [-10], [0], [0], [0], [0]], dtype=torch.float32)
    weighted_non_smoothed_loss_fn: SupervisedLearningCriterion = \
        BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0, class_counts={0.0: 1.0, 1.0: 5.0})
    weighted_smoothed_loss_fn: SupervisedLearningCriterion = \
        BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0.1, class_counts={0.0: 1.0, 1.0: 5.0})
    non_weighted_smoothed_loss_fn: SupervisedLearningCriterion = BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0.1,
                                                                                                  class_counts=None)
    w_loss1 = weighted_non_smoothed_loss_fn(logits, smoothed_target)
    w_loss2 = weighted_smoothed_loss_fn(logits, target)
    w_loss3 = non_weighted_smoothed_loss_fn(logits, target)
    positive_class_weights = weighted_smoothed_loss_fn.get_positive_class_weights()  # type: ignore
    assert torch.isclose(w_loss1, w_loss2)
    assert not torch.isclose(w_loss2, w_loss3)
    assert torch.all(positive_class_weights == torch.tensor([[0.2]]))


def test_weighted_binary_cross_entropy_loss_multi_target() -> None:
    target = torch.tensor([[[1], [0]], [[1], [0]], [[0], [0]]], dtype=torch.float32)
    smoothed_target = torch.tensor([[[0.9], [0.1]], [[0.9], [0.1]], [[0.1], [0.1]]], dtype=torch.float32)
    logits = torch.tensor([[[-10], [1]], [[-10], [1]], [[10], [0]]], dtype=torch.float32)
    weighted_non_smoothed_loss_fn: SupervisedLearningCriterion = \
        BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0, class_counts={1.0: 2, 0.0: 4})
    weighted_smoothed_loss_fn: SupervisedLearningCriterion = \
        BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0.1, class_counts={1.0: 2, 0.0: 4})
    non_weighted_smoothed_loss_fn: SupervisedLearningCriterion = \
        BinaryCrossEntropyWithLogitsLoss(smoothing_eps=0.1, class_counts=None)
    w_loss1 = weighted_non_smoothed_loss_fn(logits, smoothed_target)
    w_loss2 = weighted_smoothed_loss_fn(logits, target)
    w_loss3 = non_weighted_smoothed_loss_fn(logits, target)
    positive_class_weights = weighted_smoothed_loss_fn.get_positive_class_weights()  # type: ignore
    assert torch.isclose(w_loss1, w_loss2)
    assert not torch.isclose(w_loss2, w_loss3)
    assert torch.all(positive_class_weights == torch.tensor(2))


class ToyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(50, 50)
        self.relu1 = torch.nn.ReLU()
        self.dout = torch.nn.Dropout(0.2)
        self.out = torch.nn.Linear(50, 2)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        logits = self.out(dout)

        return logits
