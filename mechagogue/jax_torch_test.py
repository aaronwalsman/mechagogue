import time

def test_jax():
    import jax.numpy as jnp
    import jax.random as jrng
    key = jrng.key(1234)
    
    t0 = time.time()
    for i in range(1000):
        key, key_a, key_b, key_c, key_d, key_e, key_f, key_g = jrng.split(
            key, 8)
        a = jrng.uniform(key_a, shape=(3, 128))
        b = jrng.uniform(key_b, shape=(128,256))
        c = jrng.uniform(key_c, shape=(256,512))
        d = jrng.uniform(key_d, shape=(512,512))
        e = jrng.uniform(key_e, shape=(512,512))
        f = jrng.uniform(key_f, shape=(512,512))
        g = jrng.uniform(key_g, shape=(512,8))
        logits = a @ b @ c @ d @ e @ f @ g
    
    logits.block_until_ready()
    t1 = time.time()
    print(t1-t0)
    
def test_torch():
    import torch
    torch.random.manual_seed(1234)
    device='mps'
    
    t0 = time.time()
    for i in range(1000):
        a = torch.rand((3,128), device=device)
        b = torch.rand((128,256), device=device)
        c = torch.rand((256,512), device=device)
        d = torch.rand((512,512), device=device)
        e = torch.rand((512,512), device=device)
        f = torch.rand((512,512), device=device)
        g = torch.rand((512,8), device=device)
        logits = a @ b @ c @ d @ e @ f @ g
    
    _ = logits.cpu()
    t1 = time.time()
    print(t1-t0)
    
if __name__ == '__main__':
    #test_torch()
    test_jax()
