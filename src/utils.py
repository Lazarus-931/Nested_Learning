import optax
import flax.nnx as nnx
import flax.linen as nn
import jax
from typing import NamedTuple, Callable
import jax.numpy as jnp
import optax





def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


# Vanilla

# Multiscale momentum/memory muon (continuous memory system in optimizers)

class MuonState(NamedTuple):
    M1: float
    M2: float
    V: float
    k: float
    t: float

class MultiScaleMomentumMuon(nnx.Optimizer):
    """Multiscale momentum optimizer from pg 29 (7.2)"""

    def __init__(self,
                 initial_weights: int,
                 params,
                 objective_loss: float,
                 newton_schulz_steps: int = 10,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 beta3: float = 0.9,
                 alpha: float = 0.1,
                 momentum=0.95,
                 decay_rate=0,
                 lr: float = 1e-3,
                 frequency: int = 10,
                 ):

        def init_fn(params):
            return MuonState(
                M1 = jax.tree.map(jnp.zeros_like, params),
                M2 = jax.tree_map(jnp.zeros_like, params),
                V = jax.tree_map(jnp.zeros_like, params),
                k = 0,
                t = 0
            )


        defaults = dict(lr=lr, momentum=momentum, decay_rate=decay_rate)
        super().__init__(params, defaults)

    def __call__(self, x) -> optax.GradientTransformation:
        m1 = MuonState.M1.fla
        for _ in range(MuonState.k):
            pass



