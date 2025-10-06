'''
Lightweight alternative to MNIST dattaset for testing.

Creates synthetic 7x7 images of digits 0-9 with optional shifts, noise,
and color variations.
'''

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

byte_digits = [
b'''
 XXX 
X   X
X   X
X   X
 XXX 
''',

b'''
  X  
 XX  
X X  
  X  
XXXXX
''',

b'''
 XXX 
X   X
  XX 
 X   
XXXXX
''',

b'''
XXXX 
    X
  XX 
    X
XXXX 
''',

b'''
X   X
X   X
XXXXX
    X
    X
''',

b'''
XXXXX
X    
XXXX 
    X
XXXX 
''',

b'''
 XX  
X    
XXXX 
X   X
 XXX 
''',

b'''
XXXXX
   X 
  X  
 X   
X    
''',

b'''
 XXX 
X   X
 XXX 
X   X
 XXX 
''',

b'''
 XXX 
X   X
 XXXX
    X
  XX 
''',
]

def bytes_to_bool_array(digit):
    raw = digit.replace(b'\n', b'')
    array = np.frombuffer(raw, dtype=np.uint8)
    array = array != np.frombuffer(b' ', dtype=np.uint8)
    array = jnp.array(array).reshape(5,5)
    return array

bool_digits = jnp.stack(
    [bytes_to_bool_array(byte_digit) for byte_digit in byte_digits])

def array_to_bytes(digit):
    return '\n'.join([''.join(' X'[d] for d in line) for line in digit])

def make_image(
    key,
    digit,
    include_shifts=False,
    noise=0.,
    random_colors=False,
    dtype=jnp.bfloat16,
):
    shift_key, noise_key, color_key = jrng.split(key, 3)
    
    bool_digit = bool_digits[digit]
    padded_digit = jnp.zeros((7,7), dtype=jnp.bool)
    if include_shifts:
        start_y, start_x = jrng.randint(shift_key, (2,), 0, 3)
    else:
        start_y, start_x = 1, 1
    
    y, x = jnp.meshgrid(jnp.arange(5), jnp.arange(5), indexing='ij')
    y = y + start_y
    x = x + start_x
    padded_digit = padded_digit.at[y, x].set(bool_digit)
    
    if noise:
        noise_mask = jrng.bernoulli(noise_key, p=noise, shape=(7,7))
        padded_digit = padded_digit != noise_mask
    
    if random_colors:
        foreground_color, background_color = jrng.uniform(
            color_key, shape=(2,3), minval=-1, maxval=1, dtype=dtype)
    else:
        foreground_color, background_color = jnp.array([1,-1], dtype=dtype)
    
    padded_digit = jnp.where(
        padded_digit[...,None], foreground_color, background_color)
    
    return padded_digit

def make_dataset(
    key,
    n,
    include_shifts=False,
    noise=0.,
    random_colors=False,
    dtype=jnp.bfloat16,
):
    y_key, x_key = jrng.split(key)
    y = jrng.randint(y_key, minval=0, maxval=10, shape=(n,))
    
    x = jax.vmap(make_image, in_axes=(0,0,None,None,None,None))(
        jrng.split(x_key, n),
        y,
        include_shifts,
        noise,
        random_colors,
        dtype,
    )
    
    return x, y

key = jrng.key(0x123456789)

key, maxnist_10_key = jrng.split(key)
maxnist_10_x, maxnist_10_y = make_dataset(
    maxnist_10_key, 60000,
    include_shifts=True,
    noise=0.025,
    random_colors=False,
)
maxnist_10_x_train, maxnist_10_x_test = (
    maxnist_10_x[:50000], maxnist_10_x[50000:])
maxnist_10_y_train, maxnist_10_y_test = (
    maxnist_10_y[:50000], maxnist_10_y[50000:])

def make_multidigit_dataset(key, digits, n, *args, **kwargs):
    keys = jrng.split(key, digits)
    xs, ys = [], []
    for d, k in enumerate(keys):
        x, y = make_dataset(k, n, *args, **kwargs)
        xs.append(x)
        ys.append(y * 10**(digits-d-1))
    
    x = jnp.concatenate(xs, axis=2)
    y = sum(ys)
    
    return x, y
