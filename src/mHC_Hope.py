
import jax
import optax
import flax
import optax.schedules as sche
import flax.linen as nn
import flax.nnx as nnx
import jax.numpy as jnp
import einops



class Vanilla_Optimizer(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params


    def __call__(self, gradient, state):
        return nn.grad(gradient, state)



# A meta-learned optimizer, taking the aspects of mHC? Possible?
class mHC_Optimizer(nn.Module):
    def __init__(self):
        self.streams = 4
        self.recurrent = nn.recurrent

    def dept(self, x):



    def width(self, x):




    def __call__(x, gradient, state):




# mHC based mHC endured Hyper Connection layer instead of a single layer for Nested Learning?



