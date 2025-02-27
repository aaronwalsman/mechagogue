import jax.random as jrng

def make_example_data(
    key,
    num_examples,
    num_classes=16,
    in_channels=256,
    data_noise=0.,
):
    key, class_key = jrng.split(key)
    x_classes = jrng.normal(class_key, shape=(num_classes, in_channels))

    key, y_key = jrng.split(key)
    y = jrng.randint(y_key, minval=0, maxval=num_classes, shape=num_examples)

    x = x_classes[y]
    key, noise_key = jrng.split(key)
    x = x + jrng.normal(
        noise_key, shape=(num_examples, in_channels)) * data_noise

    return x, y
