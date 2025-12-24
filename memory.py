from dataclasses import dataclass
import flax
import jax
import flax as nn
import flax.linen as nnx
import jax.numpy as jnp


@dataclass
class MemoryLayerArgs:
    dim: int = 512,
    hidden_mult: int = 4,
    num_layers: int = 2,

class MemoryMLP(nnx.Module):
    def __init__(self, dim: int, num_layers: int, hidden_mult: int):
        super().__init__()
        hidden = num_layers * hidden_mult
        self.num_layers = num_layers
        self.dim = dim
        self.variable = None

    def __call__(self, x):
        h = x
        for i in range(self.num_layers):
            out_dim = self.dim if i == self.num_layers - 1 else self.hidden
            h = nnx.Dense(out_dim)(h)
            if i < self.num_layers - 1:
                h = nnx.silu(h)
        return h


class Memory(nnx.Module):
    def __init__(self, config: MemoryLayerArgs, num_persistent: int = 16):
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


    def loss_function(self, k_t: int, v_t: int, memory_params) -> jnp.ndarray:
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

    def update_memory(self, memory_params, momentum_state, x_t, eta_t, theta_t):
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
        alpha_logit = self.alpha_m(x_t)
        alpha_sig = jax.nn.sigmoid(alpha_logit)

        k_t = x_t @ self.W_K
        v_t = x_t @ self.W_V

        loss_fn = lambda params: jnp.sum((self.memory_forward(params, k_t) - v_t) ** 2)

        gradient = jax.grad(loss_fn)(memory_params)

        new_momentum_state = (eta_t * momentum_state) - (theta_t * gradient)

        new_memory_params = jax.tree_map(
            lambda s, m: s + ((1 - alpha_sig) * m),
            memory_params,
            new_momentum_state
        )

        return new_memory_params, new_momentum_state


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




