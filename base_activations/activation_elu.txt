@jax.jit
def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array:
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Args:
    x : input array
    alpha : scalar or array of alpha values (default: 1.0)

  Returns:
    An array.

  See also:
    :func:`selu`
  """
  numpy_util.check_arraylike("elu", x)
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr > 0,
                   x_arr,
                   alpha * jnp.expm1(jnp.where(x_arr > 0, 0., x_arr)))
