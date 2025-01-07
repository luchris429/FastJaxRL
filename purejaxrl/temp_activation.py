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


import jax
import jax.numpy as jnp

def custom_activation(x, alpha=10.0): # alpha controls the steepness of the transition
  """
  A smooth approximation of ReLU with a controllable steepness.
  """
  return jnp.where(x > 0, x, alpha * jnp.expm1(x))

