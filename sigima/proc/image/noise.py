# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Noise addition computation module.
----------------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y...

# Note:
# ----
# - All `guidata.dataset.DataSet` parameter classes must also be imported in the
#   `sigima.params` module.
# - All functions decorated by `computation_function` must be imported in the upper
#   level `sigima.proc.image` module.

from __future__ import annotations

from sigima.objects.base import (
    BaseNormalRandomParam,
    BasePoissonRandomParam,
    BaseUniformRandomParam,
)
from sigima.objects.image import (
    ImageObj,
    NormalRandom2DParam,
    PoissonRandom2DParam,
    UniformRandom2DParam,
    create_image_from_param,
)
from sigima.proc.base import dst_1_to_1
from sigima.proc.decorator import computation_function
from sigima.proc.image.arithmetic import addition

__all__ = ["add_gaussian_noise"]


@computation_function()
def add_gaussian_noise(src: ImageObj, p: BaseNormalRandomParam) -> ImageObj:
    """Add Gaussian (normal) noise to the input image.

    Args:
        src: Source image.
        p: Parameters.

    Returns:
        Result image object.
    """
    param = NormalRandom2DParam.create(seed=p.seed, mu=p.mu, sigma=p.sigma)
    assert src.data is not None
    shape = src.data.shape
    param.height = shape[0]
    param.width = shape[1]
    param.dtype = src.data.dtype
    noise = create_image_from_param(param)
    dst = dst_1_to_1(src, "add_gaussian_noise", f"mu={p.mu},sigma={p.sigma}")
    dst.data = addition([dst, noise]).data
    return dst


@computation_function()
def add_poisson_noise(src: ImageObj, p: BasePoissonRandomParam) -> ImageObj:
    """Add Poisson noise to the input image.

    Args:
        src: Source image.
        p: Parameters.

    Returns:
        Result image object.
    """
    param = PoissonRandom2DParam.create(seed=p.seed, lam=p.lam)
    assert src.data is not None
    shape = src.data.shape
    param.height = shape[0]
    param.width = shape[1]
    param.dtype = src.data.dtype
    noise = create_image_from_param(param)
    dst = dst_1_to_1(src, "add_poisson_noise", f"lam={p.lam}")
    dst.data = addition([dst, noise]).data
    return dst


@computation_function()
def add_uniform_noise(src: ImageObj, p: BaseUniformRandomParam) -> ImageObj:
    """Add uniform noise to the input image.

    Args:
        src: Source image.
        p: Parameters.

    Returns:
        Result image object.
    """
    param = UniformRandom2DParam.create(seed=p.seed, vmin=p.vmin, vmax=p.vmax)
    assert src.data is not None
    shape = src.data.shape
    param.height = shape[0]
    param.width = shape[1]
    param.dtype = src.data.dtype
    noise = create_image_from_param(param)
    dst = dst_1_to_1(src, "add_uniform_noise", f"low={p.vmin}, high={p.vmax}")
    dst.data = addition([dst, noise]).data
    return dst
