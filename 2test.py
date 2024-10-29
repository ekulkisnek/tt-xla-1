from jax import grad, jit, vmap
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import jax
import os
import sys
import jax._src.xla_bridge as xb

# Register cpu and tt plugin. tt plugin is registered with higher priority; so
# program will execute on tt device if not specified otherwise. 
def initialize():
  backend = "tt"
  path = os.path.join(os.path.dirname(__file__), "build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

  print("Loading tt_pjrt C API plugin", file=sys.stderr)
  xb.discover_pjrt_plugins()

  plugin = xb.register_plugin('tt', priority=500, library_path=path, options=None)
  print("Loaded", file=sys.stderr)
  jax.config.update("jax_platforms", "tt,cpu")


# Create random inputs (weights) on cpu and move them to tt device if requested.
def random_input_tensor(shape, key=42, on_device=False):
  
  device_cpu = jax.devices('cpu')[0]
  with jax.default_device(device_cpu):
    tensor = jax.random.uniform(jax.random.PRNGKey(key), shape=shape)

  #def random_input(shape, key):
   # return jax.random.uniform(key, shape=shape)
  
  #print(jax.local_devices()[0])
  #key = jax.device_put(key, jax.local_devices()[0])
  #jax_key = jax.random.PRNGKey(key)
  #jitted_tensor_creator = jax.jit(random_input, static_argnums=[0], backend='cpu')
  #tensor = jitted_tensor_creator(shape, jax_key)
  if on_device:
    tensor = jax.device_put(tensor, jax.devices()[0])
  return tensor

"""def test_gradient():
  initialize()

  def f(A):
    return (A ** 2).sum()

  A = random_input_tensor((2,2))
  print(A)
  #df_dA = jax.grad(f)(A)
  #print(df_dA[0, 0], df_dA[0, 1], df_dA[1, 2])

def test_add():
  initialize()

  def module_add(a, b):
    return a + b
  
  #a = jnp.array([[1, 5., 3.], [1, 2, 3]])
  a = random_input_tensor((4, 4))
  #print(module_add(a, a))
  print(jnp.sum(a))

def test_max():
  initialize()

  def module_max(a):
    return jnp.max(a)
  
  a = jnp.array([1., 2., 1.5])
  print(module_max(a))"""

def test_select():
  initialize()
  #def mod_select(a, b, c, d, e):
  def mod_select(pred, c, d):
    #pred = a > b
    #pred2 = c > d
    #pred3 = pred & pred2
    #return pred3 + e
    return jax.lax.select(pred, c, d)
#%2 = a, %4 = b, %6 = 
  a = jnp.array([[2., 3.]], dtype=jnp.bfloat16) #random_input_tensor((1, 2), key=0)
  b = jnp.array([[1., 3.]], dtype=jnp.bfloat16) #random_input_tensor((1, 2), key=1)
  #c = jnp.array([[5., 6.]]) #random_input_tensor((1, 2), key=2)
  #d = jnp.array([[1., 2.]]) #random_input_tensor((1, 2), key=3)
  #e = jnp.array([[1., 2.]]) #random_input_tensor((1, 2), key=3)
  graph = jax.jit(mod_select)
  #f = graph(a, b, c, d, e)
  #print(a)
  #print(b)
  #print(c)
  #print(d)
  #print(e)
  #print(f)
  pred = jnp.array([[True, False]])
  f = graph(pred, a, b)
  print(f)

def test_xor():
  initialize()
  def mod_xor(a, b, c, d, e, f):
    pred1 = a < b
    pred2 = c < d
    out1 = pred1 & pred2
    #out1 = jnp.bitwise_or(a, b)
    out2 = out1 + e
    return out2
    #return jnp.bitwise_or(a, b)

  a = jnp.array([[1., 2.]])
  b = jnp.array([[3., 4.]])
  c = jnp.array([[1., 2.]])
  d = jnp.array([[1., 2.]])
  e = jnp.array([[5., 6.]])
  f = jnp.array([[1., 2.]])
  graph = jax.jit(mod_xor)

  print(graph(a, b, c, d, e, f))

def test_funcArg():
  initialize()
  device_cpu = jax.devices('cpu')[0]
  with jax.default_device(device_cpu):
    a = jnp.array([[True, False]])
    b = jnp.array([[False, False]])

  def mod_add(a, b):
    return jnp.add(a, b)
  print(jnp.add(a, b))
  #print(jax.jit(mod_add).lower(a, b).as_text())

def test_compare():
  initialize()
  def mod_compare(a, b, c, d, e, f):
    pred1 = a > b
    pred2 = c > d
    pred3 = pred1 & pred2
    return jnp.logical_not(pred3)
    #return jnp.logical_not(pred3) + e + f

  """a = jnp.array([[3., 2.]])
  b = jnp.array([[1., 2.]])
  c = jnp.array([[1., 2.]])

  graph = jax.jit(mod_compare)
  print(graph(a, b, c))"""
  def mod_not(a):
    return jnp.logical_not(a)

  #device_cpu = jax.devices('cpu')[0]
  #with jax.default_device(device_cpu):
  """a = jnp.array([[3., 2]])
  b = jnp.array([[2., 3.]]) # T F
  c = jnp.array([[7., 8.]])
  d = jnp.array([[6., 7.]]) # T F
  e = jnp.array([[9., 10.]])
  f = jnp.array([[6., 3.]])
  g = jnp.array([[False, True]])
  #h = jnp.array([[False, True]])
  #e = True
  #b = jnp.array([[1, 2]])
  #print(jnp.bitwise_and(d, d))
  #c = jax.lax.convert_element_type(a, jnp.float32)
  graph = jax.jit(mod_compare)
  out = graph(a, b, c, d, e, f)
  print(f"out: {out}")
  print(f"converted: {out + e + f}")
  print(g)
  print(jnp.logical_not(g))"""
  #print(e)
  #print(jnp.logical_not(e))
  #print(a + b)
  #print(jnp.add(a, b))
  e = True
  print(e)
  #print(jnp.logical_and(g, g))
  #graph = jax.jit(mod_not)
  #print(f"!g {graph(g)}")
  #print(f"!h {graph(h)}")
  print(jnp.logical_not(e))
  #print (jnp.logical_not(a))
  x = jnp.array([[1, 2]])
  print(jnp.bitwise_and(x, x))

def test_finite():
  initialize()
  def mod_finite(a, b):
    c = jnp.isfinite(a)
    return c + b

  a = jnp.array([[2., -2, 1, 4], [1, 2., 3, 4], [5., 6, 7, 4], [1, 2, 3, 4]])
  b = jnp.array([[1., 2, 3, 3], [4, 5, 6., 6.], [7., 8., 9., 9.], [1, 2, 3, 4]])
  #b = jnp.array([[1., 2, 3], [4, 5, 6.], [7., 8., 9.]], dtype=jnp.bfloat16)
  #b = jnp.array([[1., 2, 3, 4, 5], [4, 5, 6., 7., 8.], [7., 8., 9., 1, 2], [4, 5, 6., 7., 8.], [7., 8., 9., 1, 2]])
  graph = jax.jit(mod_finite)
  print(a)
  print(graph(a, b))

def test_logical():
  initialize()
  def mod_add(a, b):
    return jnp.logical_and(a, a) + b
  a = jnp.array([[True, False]])
  b = jnp.array([[1., 2.]])
  graph = jax.jit(mod_add)
  print(graph(a, b))

def test_reduce():
  initialize()
  def module_reduce_sum(a, b):
    c = a < b
    #return c
    return jnp.max(c)

  a = jnp.array([[1., 2.], [3, 4.]])
  b = jnp.array([[2., 3.], [5, 4.]])
  #print ( a < b)
  graph = jax.jit(module_reduce_sum)
  print(graph(a, b))

def test_constant():
  initialize()
  def mod_constant():
    a = jnp.array([[True]])
    return a

  graph=jax.jit(mod_constant)
  print(graph())

def test_add_const():
  initialize()
  def mod_add(a, b):
    #c = jnp.array([[0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0.0, 1., 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    c = jnp.array([[0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 31, 32, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8], [0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 31, 32, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8 ]])
    return c

  a = jnp.array([[0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9]])
  b = jnp.array([[0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9, 0.0, 1., 2., 3., 4, 5, 6, 7, 8, 9]])
  graph = jax.jit(mod_add)
  print(graph(a, b))

def test_min():
  initialize()
  def func_min(a, b):
    return jnp.maximum(a, b)

  def func_clip(min, a, max):
    min1 = jnp.array([2., 2., 2., 2.])
    max1 = jnp.array([3., 3., 3., 3.])
    return jax.lax.clamp(min1, a, max1)
  
  a = jnp.array([1., 2., 3, 4])
  b = jnp.array([4., 3, 2., 1.])
  min = jnp.array([2., 2., 2., 2.])
  max = jnp.array([3., 3., 3., 3.])
  graph = jax.jit(func_min)
  graph2 = jax.jit(func_clip)
  print(graph2(min, a, max))


if __name__ == "__main__":
  test_min()
  #test_constant()
  #test_finite()
  #test_add_const()
  #test_gradient()
  #test_add()
  #test_max()
  #test_select()
  #test_xor()
  #test_funcArg()
  #test_compare()
  #test_logical()
  #test_reduce()
