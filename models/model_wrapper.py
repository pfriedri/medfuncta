import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange


def exists(val):
    return val is not None


class ModelWrapper(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.data_type = args.data_type

        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

        if self.data_type == 'img':
            self.width = args.data_size[1]
            self.height = args.data_size[2]

            mgrid = self.shape_to_coords((self.width, self.height))
            mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        elif self.data_type == 'img3d':
            self.width = args.data_size[1]
            self.height = args.data_size[2]
            self.depth = args.data_size[3]

            mgrid = self.shape_to_coords((self.width, self.height, self.depth))
            mgrid = rearrange(mgrid, 'h w d c -> (h w d) c')

        elif self.data_type == 'timeseries':
            self.length = args.data_size[-1]
            mgrid = self.shape_to_coords([self.length])

        else:
            raise NotImplementedError()

        self.register_buffer('grid', mgrid)

    def coord_init(self):
        self.sampled_coord = None
        self.sampled_index = None
        self.gradncp_coord = None
        self.gradncp_index = None

    def get_batch_coords(self, x=None):
        if x is None:
            meta_batch_size = 1
        else:
            meta_batch_size = x.size(0)

        # batch of coordinates
        if self.sampled_coord is None and self.gradncp_coord is None:
            coords = self.grid
        elif self.gradncp_coord is not None:
            return self.gradncp_coord, meta_batch_size
        else:
            coords = self.sampled_coord
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        return coords, meta_batch_size

    def shape_to_coords(self, spatial_dims):
        coords = []
        for i in range(len(spatial_dims)):
            coords.append(torch.linspace(-1.0, 1.0, spatial_dims[i]))
        return torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)

    def sample_coordinates(self, sample_type, data):
        if sample_type == 'random':
            self.random_sample()
        elif sample_type == 'gradncp':
            if random.random() < 0.5:
                self.gradncp(data)
            else:
                self.random_sample()
        else:
            raise NotImplementedError()

    def gradncp(self, x):
        ratio = self.args.data_ratio
        meta_batch_size = x.size(0)
        coords = self.grid
        coords = coords.clone().detach()[None, ...].repeat((meta_batch_size,) + (1,) * len(coords.shape))
        coords = coords.to(self.args.device)
        with torch.no_grad():
            out, feature = self.model(coords, get_features=True)

        if self.data_type == 'img':
            out = rearrange(out, 'b hw c -> b c hw')
            feature = rearrange(feature, 'b hw f -> b f hw')
            x = rearrange(x, 'b c h w -> b c (h w)')
        elif self.data_type == 'img3d':
            out = rearrange(out, 'b hwd c -> b c hwd')
            feature = rearrange(feature, 'b hwd f -> b f hwd')
            x = rearrange(x, 'b c h w d -> b c (h w d)')
        elif self.data_type == 'timeseries':
            out = rearrange(out, 'b l c -> b c l')
            feature = rearrange(feature, 'b l f -> b f l')
        else:
            raise NotImplementedError()

        error = x - out

        gradient = -1 * feature.unsqueeze(dim=1) * error.unsqueeze(dim=2)
        gradient_bias = -1 * error.unsqueeze(dim=2)
        gradient = torch.cat([gradient, gradient_bias], dim=2)
        gradient = rearrange(gradient, 'b c f hw -> b (c f) hw')
        gradient_norm = torch.norm(gradient, dim=1)

        coords_len = gradient_norm.size(1)

        self.gradncp_index = torch.sort(gradient_norm, dim=1, descending=True)[1][:, :int(coords_len * ratio)]
        self.gradncp_coord = torch.gather(coords, 1, self.gradncp_index.unsqueeze(dim=2).repeat(1, 1, self.args.in_size))
        self.gradncp_index = self.gradncp_index.unsqueeze(dim=1).repeat(1, self.args.out_size, 1)

    def random_sample(self):
        coord_size = self.grid.size(0)
        perm = torch.randperm(coord_size)
        self.sampled_index = perm[:int(self.args.data_ratio * coord_size)]
        self.sampled_coord = self.grid[self.sampled_index]
        return self.sampled_coord

    def forward(self, x=None):
        if self.data_type == 'img':
            return self.forward_img(x)
        if self.data_type == 'img3d':
            return self.forward_img3d(x)
        if self.data_type == 'timeseries':
            return self.forward_timeseries(x)
        else:
            raise NotImplementedError()

    def forward_img(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b hw c -> b c hw')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c h w -> b c (h w)')
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = rearrange(x, 'b c h w -> b c (h w)')[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)

        out = rearrange(out, 'b c (h w) -> b c h w', h=self.height, w=self.width)
        return out

    def forward_img3d(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b hwd c -> b c hwd')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = rearrange(x, 'b c h w d -> b c (h w d)')
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = rearrange(x, 'b c h w d -> b c (h w d)')[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)

        out = rearrange(out, 'b c (h w d) -> b c h w d', h=self.height, w=self.width, d=self.depth)
        return out

    def forward_timeseries(self, x):
        coords, meta_batch_size = self.get_batch_coords(x)
        coords = coords.to(self.args.device)

        out = self.model(coords)
        out = rearrange(out, 'b l c -> b c l')

        if exists(x):
            if self.sampled_coord is None and self.gradncp_coord is None:
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            elif self.gradncp_coord is not None:
                x = torch.gather(x, 2, self.gradncp_index)
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)
            else:
                x = x[:, :, self.sampled_index]
                return F.mse_loss(x.view(meta_batch_size, -1), out.reshape(meta_batch_size, -1), reduce=False).mean(dim=1)

        return out
