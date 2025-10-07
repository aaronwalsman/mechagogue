'''
Classification task utilities including cross-entropy loss and accuracy metrics.
'''

import jax
import jax.numpy as jnp
import jax.nn as jnn

def loss(pred, y, mask=None):
    b,c = pred.shape
    logp = jnn.log_softmax(pred, axis=-1)
    logpy = logp[jnp.arange(b),y]
    if mask is not None:
        logpy = logpy * mask
    loss = -jnp.mean(logpy)
    return loss

def accuracy(pred, y, mask=None):
    b,c = pred.shape
    pred = jnp.argmax(pred, axis=-1)
    correct = (pred == y)
    if mask is None:
        accuracy = jnp.mean(correct)
    else:
        correct = correct * mask
        accuracy = jnp.sum(correct) / jnp.sum(mask)
    
    return accuracy
