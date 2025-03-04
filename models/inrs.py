import torch
from torch import nn

from models.layers import LatentModulatedSIRENLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentModulatedSIREN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size=256,
                 num_layers=5,
                 latent_modulation_dim=512,
                 w0=30.,
                 w0_increments=0.,
                 modulate_shift=True,
                 modulate_scale=False,
                 enable_skip_connections=True):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            is_first = i == 0
            layer_in_size = in_size if is_first else hidden_size
            layers.append(LatentModulatedSIRENLayer(in_size=layer_in_size, out_size=hidden_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_first=is_first))
            w0 += w0_increments  # Allows for layer adaptive w0s
        self.layers = nn.ModuleList(layers)
        self.last_layer = LatentModulatedSIRENLayer(in_size=hidden_size, out_size=out_size,
                                                    latent_modulation_dim=latent_modulation_dim, w0=w0,
                                                    modulate_shift=modulate_shift, modulate_scale=modulate_scale,
                                                    is_last=True)
        self.enable_skip_connections = enable_skip_connections
        self.modulations = torch.zeros(size=[latent_modulation_dim], requires_grad=True).to(device)

    def reset_modulations(self):
        self.modulations = self.modulations.detach() * 0
        self.modulations.requires_grad = True

    def forward(self, x, get_features=False):
        x = self.layers[0](x, self.modulations)
        for layer in self.layers[1:]:
            y = layer(x, self.modulations)
            if self.enable_skip_connections:
                x = x + y
            else:
                x = y
        features = x
        out = self.last_layer(features, self.modulations) + 0.5

        if get_features:
            return out, features
        else:
            return out
