import math
import torch
from torch import nn


class LatentModulatedSIRENLayer(nn.Module):
    def __init__(self, in_size, out_size, latent_modulation_dim: 512, w0=30.,
                 modulate_shift=True, modulate_scale=False, is_first=False, is_last=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_modulation_dim = latent_modulation_dim
        self.w0 = w0
        self.modulate_shift = modulate_shift
        self.modulate_scale = modulate_scale
        self.is_last = is_last

        self.linear = nn.Linear(in_size, out_size)

        if modulate_shift:
            self.modulate_shift_layer = nn.Linear(latent_modulation_dim, out_size)
        if modulate_scale:
            self.modulate_scale_layer = nn.Linear(latent_modulation_dim, out_size)

        self._init(w0, is_first)

    def _init(self, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(6.0/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, latent):
        x = self.linear(x)
        if not self.is_last:
            shift = 0.0 if not self.modulate_shift else self.modulate_shift_layer(latent)
            scale = 1.0 if not self.modulate_scale else self.modulate_scale_layer(latent)

            if self.modulate_shift:
                if len(shift.shape) == 2:
                    shift = shift.unsqueeze(dim=1)
            if self.modulate_scale:
                if len(scale.shape) == 2:
                    scale = scale.unsqueeze(dim=1)
                
            x = scale * x + shift
            x = torch.sin(self.w0 * x)
        return x
