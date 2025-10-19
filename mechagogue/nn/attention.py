'''
Attention mechanism layer with soft and hard attention modes.
'''

import jax
import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn

from mechagogue.nn.layer import make_layer

def make_attention_layer(temperature, mode='soft'):
    def forward(key, x):
        q,k,v = x
        nq, qkc = q.shape
        nkv, vc = v.shape
        
        qk = (q[:,None] * k[None,:]).sum(-1) / (temperature * qkc**0.5)
        pqk = jnn.softmax(qk, axis=1)
        
        #jax.debug.print('q {q}\nk {k}\nqk {qk}\npqk {pqk}',
        #    q=jnp.linalg.norm(q, axis=-1),
        #    k=jnp.linalg.norm(k, axis=-1),
        #    qk=qk,
        #    pqk=pqk,
        #)
        
        if mode == 'hard':
            i = jrng.choice(key, nkv, p=pqk)
            return v[i]
        
        elif mode == 'soft':
            return (pqk[...,None] * v[None,...]).sum(axis=1)
    
    return make_layer(forward=forward)
