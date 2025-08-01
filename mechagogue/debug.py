import jax

def conditional_print(cond, msg, hostage_value):
    def true_branch(_):
        jax.debug.print(msg)
        return hostage_value
    def false_branch(_):
        return hostage_value
    
    return jax.lax.cond(cond, true_branch, false_branch, operand=None)
