from dataclasses import dataclass
from typing import Any, Literal
from collections import namedtuple
from einops import einsum, reduce, rearrange
import flax
import jax
import flax as nn
import flax.linen as nnl
from flax import nnx
import jax.numpy as jnp


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

# Multi head rmsnorm
class MultiHeadRMSNorm(nnx.Module):
    def __init__(self, dim, heads, rngs):
        self.rmsnorm = nnx.RMSNorm(dim)
        self.gamma = nnx.Param(jnp.zeros((heads, 1, dim)))

    def __call__(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)


class AveragePool(nnx.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size


    def __call__(self, x, chunk_size = None):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(nnx.Module):

    def __init__(self, dim, chunk_size):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size

        self.to_attn_logits = flax.nnx.Linear(in_features=dim, out_features=dim)
        nnx.zeros_init(self.to_attn_logits.weights)
        nnx.zeros_init(self.to_attn_logits.bias)

    def __call__(self, x, chunk_size = None):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b n d -> b (n d) c', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim=-2)

        return reduce(x * attn, 'b n d -> b (n d) c', 'sum')

@dataclass
class MemoryLayerArgs:
    dim: int = 512,
    hidden_mult: int = 4,
    num_layers: int = 2,

class MemoryMLP(nnx.Module):
    def __init__(self, dim: int, num_layers: int, hidden_mult: int, chunk_size: int, heads: int):
        super().__init__()
        self.hidden = dim * hidden_mult
        self.num_layers = num_layers
        self.dim = dim
        self.variable = None
        self.heads = heads
        self.chunk_size = chunk_size

    def __call__(self, x):
        h = x
        for i in range(self.num_layers):
            out_dim = self.dim if i == self.num_layers - 1 else self.hidden
            h = nnx.Dense(out_dim)(h)
            if i < self.num_layers - 1:
                h = nnx.silu(h)
        return h


class Memory(nnx.Module):
    def __init__(self, config: MemoryLayerArgs, num_persistent: int = 16, chunk_size: int):
        super().__init__()

        # Store config
        self.config = config
        self.num_persistent = num_persistent

        # Create memory MLP with correct parameters from config
        self.memory_mlp = MemoryMLP(
            dim=config.dim,
            num_layers=config.num_layers,
            hidden_mult=config.hidden_mult
        )

        # Create learned hyperparameter networks
        self.eta_net = nnx.Dense(
            features=1,
            bias_init=nnx.initializers.constant(2.2)  # sigmoid(2.2) ≈ 0.9
        )
        self.theta_net = nnx.Dense(
            features=1,
            bias_init=nnx.initializers.constant(-4.6)  # softplus(-4.6) ≈ 0.01
        )
        self.alpha_net = nnx.Dense(
            features=1,
            bias_init=nnx.initializers.constant(-2.2)  # sigmoid(-2.2) ≈ 0.1
        )

        # Create projection matrices
        self.W_K = nnx.Dense(features=config.dim, use_bias=False)
        self.W_V = nnx.Dense(features=config.dim, use_bias=False)
        self.W_Q = nnx.Dense(features=config.dim, use_bias=False)

        # Create 1D convolutions for K/V/Q
        self.conv_k = nnx.Conv(
            features=config.dim,
            kernel_size=(4,),
            feature_group_count=config.dim,  # Depthwise
            padding='SAME'
        )
        self.conv_v = nnx.Conv(
            features=config.dim,
            kernel_size=(4,),
            feature_group_count=config.dim,
            padding='SAME'
        )
        self.conv_q = nnx.Conv(
            features=config.dim,
            kernel_size=(4,),
            feature_group_count=config.dim,
            padding='SAME'
        )

        # Create persistent memory parameters
        self.persistent_memory = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                jax.random.PRNGKey(0),
                (num_persistent, config.dim)
            )
        )

        # Create output gating components
        self.gate_norm = nnx.LayerNorm(config.dim)
        self.gate_proj = nnx.Dense(config.dim)
        self.micro_chunk: int = 16
        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)


    def loss_function(self, memory_params, v_t: int, k_t: int) -> jnp.ndarray:
        """
        Associative Memory loss based on equation 12, or called Momentary Surprise.

        For a given input x_t, this calculates the loss function for the memory state (t - 1), using it to
        memorize the m


        :param k_t: x_t * learnable parameter W_K
        :param v_t: x_t * learnable parameter W_V
        :param memory_params: the forward function for the memory
        :return: loss function
        """
        prediction = self.memory_mlp.apply(memory_params, k_t)
        loss = jnp.sum((prediction - v_t) ** 2)
        return loss

    def update_memory(self, memory_params, momentum_state, x_t):
        """
        This function updates the memory state based on two factors:
            1. Previous Memory State
            2. Momentary Surprise

        Using it to learn how to memorize the mapping between key and values at test time.

        Momentary Surprise: Surprise function of between keys and values

        :param memory_params: the previous memory network weights( at t-1 )
        :param momentum_state: Previous momentum weights ( at t-1 ), also known as past surprise
        :param x_t: input (i.e token) at time t
        :param eta_t: momentum decay coefficient
        :param theta_t: learning rate
        :return: nothing, updates the memory network
        """

        eta_t = jax.nn.sigmoid(self.eta_net(x_t))
        theta_t = jax.nn.softplus(self.theta_net(x_t))


        alpha_logit = self.alpha_net(x_t)
        alpha_sig = jax.nn.sigmoid(alpha_logit)

        k_t = self.W_K(x_t)
        v_t = self.W_V(x_t)

        loss_fn = lambda params: jnp.sum((self.memory_mlp.apply(params, k_t) - v_t) ** 2)

        gradient = jax.grad(self.loss_function)(memory_params, k_t, v_t)

        new_momentum_state = (eta_t * momentum_state) - (theta_t * gradient)

        new_memory_params = jax.tree_map(
            lambda m, s: s + m,
            memory_params,
            new_momentum_state
        )

        return new_memory_params, new_momentum_state

    def update_memory_chunked(
            self,
            memory_params,
            momentum,
            x_chunk,
            use_forgetting: bool = True
    ):
        """
        Process chunk in parallel.

        Args:
            use_forgetting:
                True = Equations 13-14 (with forgetting - final design)
                False = Equations 9-10 (momentum only - ablation)
        """
        # Project to K, V
        k_chunk = jax.vmap(self.W_K)(x_chunk)
        v_chunk = jax.vmap(self.W_V)(x_chunk)  # ← Fixed

        # Compute gradients in parallel
        def compute_single_gradient(k, v):
            return jax.grad(self.loss_function)(memory_params, k, v)

        gradients = jax.vmap(compute_single_gradient)(k_chunk, v_chunk)  # ← Fixed

        # Compute hyperparameters
        etas = jax.vmap(lambda x: jax.nn.sigmoid(self.eta_net(x)))(x_chunk)
        thetas = jax.vmap(lambda x: jax.nn.softplus(self.theta_net(x)))(x_chunk)  # ← Fixed

        # Update momentum (Equation 14 - same for both modes)
        def momentum_update_op(carry, inputs):
            momentum_prev = carry
            eta, theta, grad = inputs
            momentum_new = eta * momentum_prev - theta * grad
            return momentum_new, momentum_new

        _, momentums = jax.lax.scan(
            momentum_update_op,
            momentum,
            (etas, thetas, gradients)
        )

        final_momentum = momentums[-1]

        if use_forgetting:
            alphas = jax.vmap(lambda x: jax.nn.sigmoid(self.alpha_net(x)))(x_chunk)
            final_alpha = alphas[-1]

            new_memory_params = jax.tree_map(
                lambda m, s: (1 - final_alpha) * m + s,
                memory_params,
                final_momentum
            )
        else:
            new_memory_params = jax.tree_map(
                lambda m, s: m + s,
                memory_params,
                final_momentum
            )

        return new_memory_params, final_momentum

    def __call__(self, query: jnp.ndarray) -> jnp.ndarray:
        """
        A forward pass on the query, retrieving the memory without updating the weights
        :param query: the query vector (q_t = query @ self.W_Q)
        :return:
        """

        memory_params = self.memory_mlp.variable

        output = self.memory_mlp.apply(memory_params, query)

        output = jax.lax.stop_gradient(output)

        return output

    def forgetting_mechanism(self, memory_params, momentum_state, x_t):
        """
        This function provides the right method to forget certain aspects from the titan's memory. This, while similar
        to the update method, makes sure memory does not get bloated. If alpha is 0, nothing is removed, closer it is to 1,
        the more is removed. alpha ∈ [0, 1]


        :param memory_params: This is the previous memory network weights( at t-1 )
        :param momentum_state: Previous momentum weights ( at t-1 ), also known as past surprise
        :param x_t: input (i.e token) at time t
        :param eta_t: momentum decay coefficient
        :param theta_t: learning rate
        :return: forgets memory artifacts
        """

        eta_t = jax.nn.sigmoid(self.eta_net(x_t))
        theta_t = jax.nn.softplus(self.theta_net(x_t))



        alpha_logit = self.alpha_net(x_t)
        alpha_sig = jax.nn.sigmoid(alpha_logit)

        k_t = self.W_K(x_t)
        v_t = self.W_V(x_t)

        loss_fn = lambda params: jnp.sum((self.memory_mlp.apply(params, k_t) - v_t) ** 2)

        gradient = jax.grad(self.loss_function)(memory_params, k_t, v_t)

        new_momentum_state = (eta_t * momentum_state) - (theta_t * gradient)

        new_memory_params = jax.tree_map(
            lambda s, m: ((1 - alpha_sig) * m) + s,
            memory_params,
            new_momentum_state
        )

        return new_memory_params, new_momentum_state

    def chunk_sequence(self, x: jnp.ndarray, chunk_size: int) -> list[jnp.ndarray]:
        """
        Method for context memory, chunks sequence of inputs into manageable for significant decrease in
        total memory overhead.

        :param chunk_size: represents the segment sie
        :param x: input the size of (N, dim)
        :return: chunked segments, each (C, dim)
        """

        N = x.shape[0]

        num_chunks = N // chunk_size

        segments = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            segment = x[start:end]
            segments.append(segment)

        if N % chunk_size != 0:
            segments.append(x[num_chunks * chunk_size:])

        return segments

    def retrieve_memory(self, seq, weight: dict[str, jax.Array]):
        chunk_size = self.get_chunk_size

