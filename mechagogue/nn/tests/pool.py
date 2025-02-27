import jax.numpy as jnp

from mechagogue.nn.pool import maxpool_layer, avgpool_layer

def inspect_pool():
    x = jnp.zeros((3,4,4,2))
    x = x.at[:,:,:,0].set(jnp.arange(4))
    x = x.at[:,:,:,1].set(jnp.arange(4,8)[:,None])
    x = x.at[:].multiply(jnp.arange(1,4)[:,None,None,None])
    
    _, maxpool = maxpool_layer()
    y = maxpool(x)
    #_, avgpool = avgpool_layer()
    #y = avgpool(x)
    
    for i in range(3):
        for j in range(2):
            print('----')
            print(i,j)
            print(x[i,:,:,j])
            print(y[i,:,:,j])

inspect_pool()
