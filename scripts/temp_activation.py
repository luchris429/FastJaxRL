from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import operator
import math
import numpy as np
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import deprecations
from jax._src import dtypes
from jax._src import util
from jax._src.core import AxisName
from jax._src.sharding_impls import NamedSharding, PartitionSpec as P
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention as cudnn_dot_product_attention, MaskType)
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.numpy import util as numpy_util
from jax._src.typing import Array, ArrayLike, DType
from jax._src.ops.special import logsumexp as _logsumexp
from jax._src.nn.functions import *
from jax._src.nn.functions import _softmax

class Unspecified:
  def __repr__(self):
    return "_UNSPECIFIED"
_UNSPECIFIED = Unspecified()

@jax.jit
def custom_activation(x: ArrayLike, b: ArrayLike = 4) -> Array:
  r"""Squareplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{custom_activation}(x) = \frac{x + \sqrt{x^2 + b}}{2}

  as described in https://arxiv.org/abs/2112.11687.

  Args:
    x : input array
    b : smoothness parameter
  """
  numpy_util.check_arraylike("custom_activation", x)
  numpy_util.check_arraylike("custom_activation", b)
  x = jnp.asarray(x)
  b = jnp.asarray(b)
  y = x + jnp.sqrt(jnp.square(x) + b)
  return y / 2
