from typing import Any

from flax.nnx import Module
from jax.example_libraries.optimizers import Optimizer

block_type: list[Any] = [Module, Optimizer]