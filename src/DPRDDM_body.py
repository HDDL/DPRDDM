import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np
import torch
import os
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from ema_pytorch import EMA

from accelerate import Accelerator

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def l2norm(t):
    return F.normalize(t, dim=-1)


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),

            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            learned_sinusoidal_dim=16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = x.cuda()
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape  # train_number
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 根据timesteps生成timesteps个beta，逐渐增加
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            custom_loss,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=1000,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='linear',
            p2_loss_weight_gamma=0.,
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective
        self.custom_loss = custom_loss

        assert objective in {'pred_noise',
                             'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.scale = betas[1] - betas[0]
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha的累乘
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):  # predict x_0
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        # q_{x_{t-1}|x_t} mean
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # sigma
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)  # get noise of x_{t} == model_output
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)  # x_t, t, \e_{t-1}
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):  # x:x_t, t
        preds = self.model_predictions(x, t, x_self_cond)  # get the model output(noise) model_output and x_0:form
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean  # + (0.5 * model_log_variance).exp() * noise # get x_{t-1} using the p_{\theta}{u} + sigma
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):  # 从T开始采样
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)  # gaussian noise

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='origin sampling loop time step',
                      total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        # img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample_loop_x0(self, x_start, save_folder=None, save_flag=0):  # 从T开始采样
        shape = x_start.shape
        batch, device = shape[0], self.betas.device
        z_T = torch.randn(shape, device=device)  # gaussian noise
        # sample x_T using x_0
        img = x_start * self.sqrt_alphas_cumprod[-1]  # + self.sqrt_one_minus_alphas_cumprod[-1]*z_T # 由x0得到xT
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='my sampling loop time step',
                      total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        return img

    @torch.no_grad()
    def save_x_start_from_t(self, recovered_speed_matrix, save_folder, t):
        count = 1
        if t < 9:
            t = '00' + str(t + 1)
        elif t < 99:
            t = '0' + str(t + 1)
        else:
            t = str(t + 1)
        temp_batch_size = recovered_speed_matrix.shape[0]
        recovered_speed_matrix = unnormalize_to_zero_to_one(recovered_speed_matrix)
        recovered_speed_matrix = recovered_speed_matrix.cpu().numpy()
        recovered_speed_matrix *= 100
        for j in range(0, 1):
            temp_matrix = recovered_speed_matrix[j, 0, :, :]
            if count < 10:
                file_name = save_folder + '/' + t + '_00' + str(count) + '.txt'
            elif count < 100:
                file_name = save_folder + '/' + t + '_0' + str(count) + '.txt'
            else:
                file_name = save_folder + '/' + t + '_' + str(count) + '.txt'
            np.savetxt(file_name, temp_matrix, delimiter='\t')
            count += 1

    @torch.no_grad()
    def obtain_previous_x0(self, x_noise, previous_t, save_folder):  # 从x0继续回溯一步
        b, *_, device = *x_noise.shape, x_noise.device
        for i in reversed(range(0, previous_t)):
            batched_times = torch.full((x_noise.shape[0],), i, device=x_noise.device, dtype=torch.long)
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x_noise, t=batched_times)
            self.save_x_start_from_t(x_start, save_folder, i)
            noise = torch.randn_like(x_noise) if previous_t > 0 else 0.  # no noise if t == 0
            pred_img = model_mean
            print(pred_img[0, 0, 0, 0], i)
            # + (0.5 * model_log_variance).exp() * noise  # get x_{t-1} using the p_{\theta}{u} + sigma
            x_noise = pred_img

        return pred_img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
                                                                                 0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='wrong sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, idx, t, trained_model, noise=None):

        # get the x_t from forward process
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict the noise of p_\theta{x_{t-1}|x_{t}}
        model_out = self.model(x_t, t)  # predict the noise \e_theta
        pred_x_start = self.predict_start_from_noise(x_t, t, model_out)  # predict the x_start from the x_start

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.custom_loss(x_start, pred_x_start, target, model_out, idx, t, self.sqrt_alphas_cumprod[t],
                                trained_model)
        return loss

    def forward(self, img, idx, trained_model, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size  # shape: train_number, channel, height and width
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()  # 从0到time_stemps
        img = normalize_to_neg_one_to_one(img)
        loss = self.p_losses(img, idx, t, trained_model, *args, **kwargs)
        return loss


# dataset classes

class TextDataset(Dataset):
    def __init__(self, text_file_paths, max_speed, image_size):
        self.paths = text_file_paths
        self.max_speed = max_speed
        # other processing you may want

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        speed_matrix = np.loadtxt(path, delimiter=',')
        speed_matrix = np.expand_dims(speed_matrix, 0)
        speed_matrix = speed_matrix / self.max_speed  # 归一化
        speed_matrix = torch.tensor(speed_matrix)
        speed_matrix = speed_matrix.type(torch.FloatTensor)
        speed_matrix = speed_matrix.cuda()
        return (speed_matrix, idx)


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_file_paths,
            test_file_paths,
            max_speed,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_horizontal_flip=True,
            train_lr=1e-4,
            train_num_steps=10000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_samples=25,
            train_results_folder='./data/train_results/',
            test_results_folder = './data/test_results/',
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None
    ):
        super().__init__()
        # Accelerator 并行计算和半精度setting
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        self.model = diffusion_model
        self.max_speed = max_speed
        # 和group_normalization有关
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every  # 将梯度累计起来计算，相当于计算多个batch？

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # folder and file paths
        self.train_file_paths = train_file_paths
        self.test_file_paths = test_file_paths
        self.train_results_folder = train_results_folder
        if not os.path.exists(self.train_results_folder):
            os.mkdir(self.train_results_folder)

        # dataset and dataloader 读取数据, data loader每次传回batch_size个数据
        self.ds = TextDataset(train_file_paths, self.max_speed, self.image_size)
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True,
                        drop_last=False)  # num_workers = cpu_count()
        sample_dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=False,
                               drop_last=False)  # num_worker = cpu_count()
        self.sample_dl = sample_dl
        dl = self.accelerator.prepare(dl)
        self.dl = dl
        # =cycle(dl)
        # train_data
        sample_train_data_dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=False,
                                          drop_last=False)  # num_worker = cpu_count()
        self.sample_train_data_dl = sample_train_data_dl
        # test_data
        self.test_ds = TextDataset(test_file_paths, self.max_speed, self.image_size)
        sample_test_data_dl = DataLoader(self.test_ds, batch_size=train_batch_size, shuffle=False,
                                         drop_last=False)  # num_worker = cpu_count()
        self.sample_test_data_dl = sample_test_data_dl


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.995)
        self.scheduler = ExpLR
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            # 按照移动平均更新权重
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, sample_index):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),  # 保存训练的权重
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, self.train_results_folder + 'model_' + str(sample_index) + '.pt')
        print(self.train_results_folder + 'model_' + str(sample_index) + '.pt', 'saved')

    def load(self, sample_index):
        data = torch.load(self.train_results_folder + 'model_' + str(sample_index) + '.pt')

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def save_speed_matrix(self, sample_index, speed_matrix_list):
        matrix_number = speed_matrix_list.shape[0]
        for i in range(0, matrix_number):
            temp_matrix = speed_matrix_list[i, 0, :, :]
            temp_matrix = temp_matrix.cpu().numpy()
            temp_matrix *= self.max_speed
            file_name = self.results_folder + str(sample_index) + '_' + str(i) + '.txt'
            np.savetxt(file_name, temp_matrix, delimiter='\t')

    @torch.no_grad()
    def recover_origin_speed_matrix(self, sample_index, save_folder):  # 根据x_0，x_t推出x_t-1

        count = 1
        row = 360;
        column = 200;
        split_row = self.image_size;
        split_column = self.image_size
        row_max_index = row - split_row + 1;
        column_max_index = column - split_column + 1;
        # split_number = row_max_index * column_max_index
        origin_speed_matrix = np.zeros([row, column])  # 原始图像
        origin_matrix_count = np.zeros([row, column])
        for i, data in enumerate(self.sample_dl):
            temp_batch_size = data.shape[0]
            print('sampled_file_index:', i * temp_batch_size)
            recovered_speed_matrix = self.ema.ema_model.p_sample_loop_x0(data)
            recovered_speed_matrix = recovered_speed_matrix.cpu().numpy()
            recovered_speed_matrix *= self.max_speed
            for j in range(0, temp_batch_size):
                temp_matrix = recovered_speed_matrix[j, 0, :, :]
                sr = math.ceil(count / column_max_index) - 1;
                er = sr + split_row;
                sc = count - sr * column_max_index - 1;
                ec = sc + split_column;
                origin_speed_matrix[sr:er, sc:ec] += temp_matrix
                origin_matrix_count[sr:er, sc:ec] += 1
                count += 1
        origin_speed_matrix = origin_speed_matrix / origin_matrix_count
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        file_name = save_folder + str(sample_index) + '.txt'
        np.savetxt(file_name, origin_speed_matrix, delimiter='\t')
        return origin_speed_matrix

    def train_data_remove_noise(self, sample_index, save_folder, previous_time):
        # make the directory
        if not os.path.exists(save_folder + str(sample_index) + '/'):
            os.mkdir(save_folder + str(sample_index) + '/')
        count = 1
        for i, (data, idx) in enumerate(self.sample_train_data_dl):
            data = normalize_to_neg_one_to_one(data) # normalize [-1, 1]
            temp_batch_size = data.shape[0]
            print('sampled_file_index:', i * temp_batch_size)
            recovered_speed_matrix = self.ema.ema_model.p_sample_loop_x0(data, save_folder, 1)
            #recovered_speed_matrix = self.ema.ema_model.obtain_previous_x0(data, previous_time, save_folder)
            recovered_speed_matrix = unnormalize_to_zero_to_one(recovered_speed_matrix)
            recovered_speed_matrix = recovered_speed_matrix.cpu().numpy()
            recovered_speed_matrix *= self.max_speed
            for j in range(0, temp_batch_size):
                temp_matrix = recovered_speed_matrix[j, 0, :, :]
                if count < 10:
                    file_name = save_folder + str(sample_index) + '/' + '00' + str(count) + '.txt'
                elif count < 100:
                    file_name = save_folder + str(sample_index) + '/' + '0' + str(count) + '.txt'
                else:
                    file_name = save_folder + str(sample_index) + '/' + str(count) + '.txt'
                np.savetxt(file_name, temp_matrix, delimiter='\t')
                count += 1
            break

    @torch.no_grad()
    def test_data_remove_noise(self, sample_index, save_folder, previous_time):
        if not os.path.exists(save_folder + str(sample_index) + '/'):
            os.mkdir(save_folder + str(sample_index) + '/')
        count = 1
        for i, (data, idx) in enumerate(self.sample_test_data_dl):
            data = normalize_to_neg_one_to_one(data)
            temp_batch_size = data.shape[0]
            print('sampled_file_index:', i * temp_batch_size)
            recovered_speed_matrix = self.ema.ema_model.p_sample_loop_x0(data, save_folder, 1)
            #recovered_speed_matrix = self.ema.ema_model.obtain_previous_x0(data, previous_time, save_folder)
            recovered_speed_matrix = unnormalize_to_zero_to_one(recovered_speed_matrix)
            recovered_speed_matrix = recovered_speed_matrix.cpu().numpy()
            recovered_speed_matrix *= self.max_speed
            for j in range(0, temp_batch_size):
                temp_matrix = recovered_speed_matrix[j, 0, :, :]
                if count < 10:
                    file_name = save_folder + str(sample_index) + '/' + '00' + str(count) + '.txt'
                elif count < 100:
                    file_name = save_folder + str(sample_index) + '/' + '0' + str(count) + '.txt'
                else:
                    file_name = save_folder + str(sample_index) + '/' + str(count) + '.txt'
                np.savetxt(file_name, temp_matrix, delimiter='\t')
                count += 1
            break

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                # total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    for batch_index, (data, idx) in enumerate(self.dl):
                        data = data.to(device)
                        idx = idx.to(device)
                        with self.accelerator.autocast():
                            loss = self.model(data, idx, self)
                            loss = loss / self.gradient_accumulate_every
                            # total_loss += loss.item()
                        self.accelerator.backward(loss)
                        pbar.set_description(f'loss: {loss:.4f}')

                        accelerator.wait_for_everyone()

                        self.opt.step()
                        self.opt.zero_grad()

                        accelerator.wait_for_everyone()

                        self.step += 1
                        if accelerator.is_main_process:
                            self.ema.to(device)
                            self.ema.update()

                        if self.step != 0 and self.step % self.save_and_sample_every == 0:  # 每训练一定步骤进行采样，看一下训练效果
                            self.ema.ema_model.eval()
                            with torch.no_grad():
                                sample_index = self.step // self.save_and_sample_every  # 第sample_index次采样
                                self.save(sample_index)

                        pbar.update(1)
                    self.scheduler.step()
        accelerator.print('training complete')
