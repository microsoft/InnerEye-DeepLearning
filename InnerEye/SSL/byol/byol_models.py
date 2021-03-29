from typing import Tuple, Any


from torch import Tensor as T
from torch import nn

from InnerEye.SSL.encoders import Lambda
from InnerEye.SSL.ssl_classifier_module import get_encoder_output_dim
from InnerEye.SSL.utils import create_ssl_encoder


class _MLP(nn.Module):
    """
    Fully connected layers to map between image embeddings and project space where pairs of images are compared.
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

class SSLEncoder(nn.Module):
    """
    CNN image encoder that generates fixed size BYOL image embeddings.
    Feature responses are pooled to generate a 1-D embedding vector.
    """

    def __init__(self, encoder_name: str, dataset_name: str, use_output_pooling: bool = False):
        """
        :param encoder_name: Type of the image encoder: {'resnet18', 'resnet50', 'resnet101', 'densenet121'}.
        :param dataset_name: If CIFAR dataset is specified, the initial convolution kernels are reduced to 3x3
                             to reduce information loss.
        :param use_output_pooling: If set to True, output embeddings in spatial grid are pooled to form
                             a fixed size embedding vector.
        """

        super().__init__()
        self.cnn_model = create_ssl_encoder(encoder_name=encoder_name, dataset_name=dataset_name)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.use_output_pooling = use_output_pooling

    def forward(self, x: T) -> T:
        x = self.cnn_model(x)
        x = x[-1] if isinstance(x, list) else x
        x = self.avgpool(x).view(x.size(0), -1) if self.use_output_pooling else x
        return x

    def get_output_feature_dim(self) -> int:
        return get_encoder_output_dim(self)

class SiameseArm(nn.Module):
    """
    Implements the image encoder (f), projection (g) and predictor (q) modules used in BYOL.
    """

    def __init__(self, *encoder_kwargs: Any) -> None:
        super().__init__()

        self.encoder = SSLEncoder(*encoder_kwargs)  # Encoder
        self.projector = _MLP(input_dim=self.encoder.get_output_feature_dim(), hidden_dim=2048, output_dim=128)
        self.predictor = _MLP(input_dim=self.projector.output_dim, hidden_dim=128, output_dim=128)
        self.projector_normalised = nn.Sequential(self.projector,
                                                  Lambda(lambda x: nn.functional.normalize(x, dim=-1)))

    def forward(self, x: T) -> Tuple[T, T, T]:
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
