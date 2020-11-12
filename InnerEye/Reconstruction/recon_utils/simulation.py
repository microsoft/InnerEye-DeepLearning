#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import random
from typing import List, Callable, Tuple

import param
import torch

class Ellipse(param.Parameterized):
    """
    Representation of an ellipse in a phantom such as the Shepp Logan phantom.

    Image bounding box is defined to be the range [-1:1] and coordinated and lengths are defined with respect to this
    box.
    Coordinates can be outside the box if the ellipse is to be rendered partially (or completely) outside the image.
    """

    intensity: float = param.Number(default=1.0, doc="Additive intensity of ellipse")
    major: float = param.Number(default=0.5, doc="Length of major axis")
    minor: float = param.Number(default=0.5, doc="Length of minor axis")
    horizontal_offset: float = param.Number(default=0.0, doc="Horizontal offset of ellipse center")
    vertical_offset: float = param.Number(default=0.0, doc="Vertical offset of ellipse center")
    rotation: float = param.Number(default=0.0, doc="Counter clockwise rotation of ellipse in degrees.")

    def render(self, matrix_size: int = 256) -> torch.Tensor:
        """
        Renders the ellipse as a PyTorch tensor.

        :returns PyTorch tensor rendering of ellipse
        """
        # Create the pixel grid
        ygrid, xgrid = torch.meshgrid([torch.linspace(-1, 1, matrix_size), torch.linspace(-1, 1, matrix_size)])

        # Create the offset x and y values for the grid
        x = xgrid - self.horizontal_offset
        y = ygrid - self.vertical_offset

        phi = self.rotation * math.pi / 180  # Rotation angle in radians
        cos_p = torch.cos(torch.tensor([phi]))
        sin_p = torch.sin(torch.tensor([phi]))

        # Find the pixels within the ellipse
        loc = (((x * cos_p + y * sin_p) ** 2) / self.major ** 2 + ((y * cos_p - x * sin_p) ** 2) / self.minor ** 2) <= 1

        rendering = torch.zeros((matrix_size, matrix_size), dtype=torch.float)
        rendering[loc] = self.intensity

        return rendering


EllipsesList = List[Ellipse]


def modified_shepp_logan_definition() -> EllipsesList:
    """
    Modified version of Shepp & Logan's head phantom, adjusted to improve contrast.

    Taken from Toft. "The radon transform-theory and implementation." PhD Thesis, Technical University of Denmark, 1996

    :returns List of Ellipses
    """
    return [
        Ellipse(intensity=1, major=.69, minor=.92, horizontal_offset=0, vertical_offset=0, rotation=0),
        Ellipse(intensity=-.80, major=.6624, minor=.8740, horizontal_offset=0, vertical_offset=-.0184, rotation=0),
        Ellipse(intensity=-.20, major=.1100, minor=.3100, horizontal_offset=.22, vertical_offset=0, rotation=-18),
        Ellipse(intensity=-.20, major=.1600, minor=.4100, horizontal_offset=-.22, vertical_offset=0, rotation=18),
        Ellipse(intensity=.10, major=.2100, minor=.2500, horizontal_offset=0, vertical_offset=.35, rotation=0),
        Ellipse(intensity=.10, major=.0460, minor=.0460, horizontal_offset=0, vertical_offset=.1, rotation=0),
        Ellipse(intensity=.10, major=.0460, minor=.0460, horizontal_offset=0, vertical_offset=-.1, rotation=0),
        Ellipse(intensity=.10, major=.0460, minor=.0230, horizontal_offset=-.08, vertical_offset=-.605, rotation=0),
        Ellipse(intensity=.10, major=.0230, minor=.0230, horizontal_offset=0, vertical_offset=-.606, rotation=0),
        Ellipse(intensity=.10, major=.0230, minor=.0460, horizontal_offset=.06, vertical_offset=-.605, rotation=0)]


def shepp_logan_definition() -> EllipsesList:
    """
    Standard (original) head phantom elipses for the Shepp-Logan phantom.
    Taken from Shepp & Logan "The Fourier Reconstruction of a Head Section" IEEE Transactions on Nuclear Science.
    NS-21 (3): 21–43.

    See https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom

    :returns List of Ellipses
    """
    return [
        Ellipse(intensity=2, major=.69, minor=.92, horizontal_offset=0, vertical_offset=0, rotation=0),
        Ellipse(intensity=-.98, major=.6624, minor=.8740, horizontal_offset=0, vertical_offset=-.0184, rotation=0),
        Ellipse(intensity=-.02, major=.1100, minor=.3100, horizontal_offset=.22, vertical_offset=0, rotation=-18),
        Ellipse(intensity=-.02, major=.1600, minor=.4100, horizontal_offset=-.22, vertical_offset=0, rotation=18),
        Ellipse(intensity=.01, major=.2100, minor=.2500, horizontal_offset=0, vertical_offset=.35, rotation=0),
        Ellipse(intensity=.01, major=.0460, minor=.0460, horizontal_offset=0, vertical_offset=.1, rotation=0),
        Ellipse(intensity=.02, major=.0460, minor=.0460, horizontal_offset=0, vertical_offset=-.1, rotation=0),
        Ellipse(intensity=.01, major=.0460, minor=.0230, horizontal_offset=-.08, vertical_offset=-.605, rotation=0),
        Ellipse(intensity=.01, major=.0230, minor=.0230, horizontal_offset=0, vertical_offset=-.606, rotation=0),
        Ellipse(intensity=.01, major=.0230, minor=.0460, horizontal_offset=.06, vertical_offset=-.605, rotation=0)]


def random_ellipse() -> Ellipse:
    """
    Generates random ellipse parameters.

    :return ellipse parameters
    """

    s = random.random()
    a = random.random()
    b = random.random()
    x0 = random.random() * (1 - max(a, b))
    y0 = random.random() * (1 - max(a, b))
    phi = (random.random() - 0.5) * 360.0
    return Ellipse(intensity=s, major=a, minor=b, horizontal_offset=x0, vertical_offset=y0, rotation=phi)


def random_phantom_definition(ellipses: int = 10) -> EllipsesList:
    """
    Generates a list of random ellipses.

    :param ellipses: Number of ellipses
    :returns list of ellipses parameters
    """

    return [random_ellipse() for _ in range(ellipses)]


def phantom(matrix_size: int = 256,
            phantom_type: Callable[[], EllipsesList] = modified_shepp_logan_definition) -> torch.Tensor:
    """
    Create an ellipses based numerical phan such as the Shepp-Logan or modified Shepp-Logan phantom and returns it as
    a Pytorch tensor

    References:

    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    :param matrix_size: size of imaging matrix in pixels (default 256)
    :param phantom_type: The type of phantom to produce. Shoule be a function that return a list of ellipses. Default
    is modified_shepp_logan_definition
    :return PyTorch tensor with the phantom
    """

    ellipses = phantom_type()
    ph = torch.zeros((matrix_size, matrix_size), dtype=torch.float)

    for ellip in ellipses:
        ph += ellip.render(matrix_size=matrix_size)

    return ph


def generate_birdcage_sensitivities(matrix_size: int = 256,
                                    number_of_coils: int = 8,
                                    relative_radius: float = 1.5,
                                    rotation: float = 0.0,
                                    normalize: bool = True) -> torch.Tensor:
    """ 
    Generates a PyTorch complex tensor with birdcage coil sensitivites.

    :param matrix_size: size of imaging matrix in pixels (default ``256``)
    :param number_of_coils: Number of simulated coils (default ``8``)
    :param relative_radius: Relative radius of birdcage (default ``1.5``) with respect to field of view (FOV).
    ``1.5`` means coil centers are outside FOV.
    :param rotation: rotation of coil array expressed in radians
    :return torch.Tensor with shape [number_of_coils, matrix_size, matrix_size]

    This function is heavily inspired by:

        1. The mri_birdcage.m Matlab script in Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/
        2. generate_birdcage_sensitivities function from ISMRMRD Python tools (
        https://github.com/ismrmrd/ismrmrd-python-tools)
    """

    out = torch.zeros((number_of_coils, matrix_size, matrix_size), dtype=torch.cfloat)
    x_range = [x - matrix_size // 2 for x in range(matrix_size)]
    y_range = [y - matrix_size // 2 for y in range(matrix_size)]
    x_coord = torch.tensor(x_range, dtype=torch.float).unsqueeze(0).repeat(matrix_size, 1) / (matrix_size // 2)
    y_coord = torch.tensor(y_range, dtype=torch.float).unsqueeze(-1).repeat(1, matrix_size) / (matrix_size // 2)

    for c in range(number_of_coils):
        coilx = relative_radius * math.cos(c * (2 * math.pi / number_of_coils) + rotation)
        coily = relative_radius * math.sin(c * (2 * math.pi / number_of_coils) + rotation)
        coil_phase = -c * (2 * math.pi / number_of_coils)
        y_co = y_coord - coily
        x_co = x_coord - coilx
        rr = torch.sqrt(x_co ** 2 + y_co ** 2)
        phi = torch.atan2(x_co, -y_co) + coil_phase
        out[c, ...] = 1.0 / rr * torch.exp(1j * phi)

    if normalize:
        rss = torch.sqrt(torch.sum(torch.abs(out * out.conj()), 0))
        out = out / rss

    return out

def generate_phase_roll(matrix_size: int = 256, rotation: float = 0.0, center: Tuple[float, float] = (0.0, 0.0), roll: float = math.pi) -> torch.Tensor:
    """
    Generates a complex tensor that represents a linear phase gradient.
    The gradient will be centered (zero phase) at relative coordinates defined by ``center``. Coordinates are in the range [-0.5:0.5].
    The phase gradient is rotated by ``rotation`` radians. The total phase roll is defined by ``roll``, which is relative to the field of
    view, i.e. it is the total amount of phase change over a distance equal to a field of view.

    :params matrix_size: Matrix size, output with will [matrix_size, matrix_size]
    :params rotation: Rotation of the phase gradient in radians
    :params center: The center (zero) phase of the phase gradient. Coordinates in range [-0.5:0.5]
    :params roll: Total phase roll over a distance equal to half of the field of view.

    :returns PyTorch tensor with dimensions [matrix_size, matrix_size]
    """

    cos_angle = torch.cos(torch.Tensor([rotation]))
    sin_angle = torch.sin(torch.Tensor([rotation]))
    rel_x = torch.Tensor([x-matrix_size // 2 for x in range(matrix_size)]).unsqueeze(0).repeat(matrix_size, 1) / (matrix_size//2) - center[0]
    rel_y = torch.Tensor([y-matrix_size // 2 for y in range(matrix_size)]).unsqueeze(-1).repeat(1, matrix_size) / (matrix_size//2) - center[1]
    x_rot = rel_x*cos_angle - rel_y*sin_angle

    return torch.cos(x_rot*roll) + 1j*torch.sin(x_rot*roll)


def generate_random_phase_roll(matrix_size: int = 256) -> torch.Tensor:
    """
    Generates a random linear phase roll. Between 0 and 4*pi of phase is applied across the field of view. 
    The center and rotation of the phase roll is random.

    :params matrix_size: Size of the matrix, output is of size [matrix_size, matrix_size]
    :returns PyTorch tensor with dimensions [matrix_size, matrix_size]
    """

    return generate_phase_roll(matrix_size=matrix_size,
                               rotation=(random.random() - 0.5)*math.pi,
                               center=(random.random() - 0.5, random.random() - 0.5),
                               roll=random.random() * math.pi * 4)
