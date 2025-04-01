import torch
import torch.nn as nn
from torchcfm.models.unet.unet import UNetModel
from torchdiffeq import odeint
from tqdm.auto import tqdm


def build_model(cfg) -> UNetModel:
    return UNetModel(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        model_channels=cfg.base_channels,
        out_channels=cfg.out_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        channel_mult=cfg.channel_mult,
        dropout=0.1,
    )


def ema(source, target, alpha):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * alpha + source_dict[key].data * (1 - alpha))


class Sampler(nn.Module):
    def __init__(self, model: nn.Module, num_steps: int = 250):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, t, x):
        shaped_t = torch.ones(x.shape[0], device=x.device) * t
        vec = self.model(args=(shaped_t, x))
        # Zero vectorfield for conditional variable
        cond_vec = torch.zeros_like(vec)

        return torch.cat([vec, cond_vec], dim=1)

    @torch.no_grad()
    def sample(self, cond):
        x0 = torch.randn_like(cond)
        x = torch.cat([x0, cond], dim=1)

        time = torch.linspace(0, 1, self.num_steps + 1, device=x.device)
        x1 = odeint(self, x, t=time, method="euler")[-1]

        # Excluding cond
        x1 = x1[:, 0]

        # Real to Categorical
        sample = torch.where(x1 > 0, 1, 0)
        return sample

    @torch.no_grad()
    def sample_n(self, cond, n, use_tqdm=False):
        samples = []
        for _ in tqdm(range(n), disable=not use_tqdm):
            samples.append(self.sample(cond))

        return torch.stack(samples, dim=1)
