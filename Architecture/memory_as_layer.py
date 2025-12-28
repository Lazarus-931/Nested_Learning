from flax.core import nn



class MemoryAsLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size



