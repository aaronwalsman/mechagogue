from mechagogue.standardize import standardize_interface

def standardize_optimizer(optimizer):
    return standardize_interface(
        optimizer,
        init=(("key", "model_state"), None),
        optimize=(("key", "grad", "model_state", "optim_state"), None),
    )
