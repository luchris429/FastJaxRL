@jax.jit
def squareplus(x: ArrayLike, b: ArrayLike = 4) -> Array:
  r"""Squareplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

  as described in https://arxiv.org/abs/2112.11687.

  Args:
    x : input array
    b : smoothness parameter
  """
  numpy_util.check_arraylike("squareplus", x)
  numpy_util.check_arraylike("squareplus", b)
  x = jnp.asarray(x)
  b = jnp.asarray(b)
  y = x + jnp.sqrt(jnp.square(x) + b)
  return y / 2
