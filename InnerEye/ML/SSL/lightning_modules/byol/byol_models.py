from typing import Any, Tuple

from torch import Tensor as T, nn

from InnerEye.ML.SSL.encoders import Lambda, SSLEncoder


class _MLP(nn.Module):
    """
    Fully connected layers to map between image embeddings and projection space where pairs of images are compared.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        :param input_dim: Input embedding feature size
        :param hidden_dim: Hidden layer size in MLP
        :param output_dim: Output projection size
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True))

    def forward(self, x: T) -> T:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    """
    Implements the image encoder (f), projection (g) and predictor (q) modules used in BYOL.
    """

    def __init__(self, *encoder_kwargs: Any) -> None:
        super().__init__()

        self.encoder = SSLEncoder(*encoder_kwargs)  # Encoder
        self.projector = _MLP(input_dim=self.encoder.get_output_feature_dim(), hidden_dim=2048, output_dim=128)
        self.predictor = _MLP(input_dim=self.projector.output_dim, hidden_dim=128, output_dim=128)

    def forward(self, x: T) -> T:
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return h

    def forward_until_predictor(self, x: T) -> T:
        y = self.encoder(x)
        return self.projector(y)
