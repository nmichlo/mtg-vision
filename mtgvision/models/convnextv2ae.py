import time
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from mtgvision.models.convnextv2 import Block, LayerNorm, trunc_normal_
from mtgvision.models.nn import AeBase


def Act():
    return nn.Mish(inplace=True)


# def Norm2d(*args, **kwargs):
#     data = kwargs.pop("data_format", None)
#     if data != "channels_first":
#         class Norm(nn.BatchNorm2d):
#             def forward(self, x):
#                 x = x.permute(0, 3, 1, 2)
#                 x = super().forward(x)
#                 x = x.permute(0, 2, 3, 1)
#                 return x
#     else:
#         class Norm(nn.BatchNorm2d):
#             pass
#     return Norm(*args, **kwargs)

def Norm2d(*args, **kwargs):
    return LayerNorm(*args, **kwargs)


def ConvBlock(**kwargs):
    return Block(norm=Norm2d, act=Act, **kwargs)


class GlobalAveragePooling(nn.Module):

    def forward(self, x):
        return x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)


class _Base(nn.Module):

    def __init__(
        self,
        image_wh: tuple[int, int] = (192, 128),
        in_chans: int = 3,
        z_size: int = 1000,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_init_scale: Optional[float] = None
    ):
        super().__init__()
        self.image_wh = image_wh
        self.in_chans = in_chans
        self.z_size = z_size
        self.depths = depths
        self.dims = dims
        self.head_init_scale = head_init_scale

        assert len(depths) == len(dims)
        iw, ih = self.get_internal_wh(image_wh)
        self.internal_wh = (iw, ih)
        self.internal_num = iw * ih

        assert z_size % self.internal_num == 0

    @property
    def internal_scale(self) -> int:
        return 4 * 2 * 2 * 2

    def get_internal_wh(self, in_wh: tuple[int, int]) -> tuple[int, int]:
        assert len(in_wh) == 2
        assert in_wh[0] % self.internal_scale == 0
        assert in_wh[1] % self.internal_scale == 0
        iw = in_wh[0] // self.internal_scale
        ih = in_wh[1] // self.internal_scale
        return iw, ih

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        # else:
        #     print(f"skipping {type(m)}")


class ConvNeXtV2Encoder(_Base):
    """ConvNeXt V2

    Args:
        in_chans: Number of input image channels. Default: 3
        z_size: Number of classes for classification head. Default: 1000
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
        head_init_scale: Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 1000,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_type: Literal['conv', 'pool+linear'] = 'conv',
        head_init_scale: Optional[float] = None,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
            head_init_scale=head_init_scale,
        )
        self.head_type = head_type

        # **DOWNSAMPLE LAYERS**
        # + stem and 3 intermediate downsampling conv layers
        # + 4 feature resolution stages, each consisting of multiple residual blocks
        # --> Bx3x192x128
        self.block0 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Norm2d(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Sequential(*[ConvBlock(dim=dims[0]) for _ in range(depths[0])]),
        )
        # --> Bx[0]x48x32
        self.block1 = nn.Sequential(
            Norm2d(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[1]) for _ in range(depths[1])]),
        )
        # --> Bx[1]x24x16
        self.block2 = nn.Sequential(
            Norm2d(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[2]) for _ in range(depths[2])]),
        )
        # --> Bx[2]x12x8
        self.block3 = nn.Sequential(
            Norm2d(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[3]) for _ in range(depths[3])]),
        )

        # **HEAD**
        # --> Bx[3]x6x4
        if head_type == 'conv':
            self.pool = nn.Sequential(
                nn.Conv2d(dims[3], z_size // (6 * 4), kernel_size=1, stride=1),
                View((-1, z_size)),
            )
            self.head = nn.Identity()
        elif head_type == 'pool+linear':
            self.pool = nn.Sequential(
                GlobalAveragePooling(),
                Norm2d(dims[3], eps=1e-6),
            )
            # --> Bx[3]
            self.head = nn.Linear(dims[3], z_size)
        else:
            raise KeyError(f"head_type={head_type} not recognized")
        # --> Bx<z_size>

        # initialize weights
        self.apply(self._init_weights)
        if self.head_init_scale is not None:
            if head_type == 'pool+linear':
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class Index(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[self.shape]


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Print(nn.Module):

    def forward(self, x):
        print(x.shape)
        return x


class ConvNeXtV2Decoder(_Base):
    """ConvNeXt V2

    Args:
        in_chans: Number of input image channels. Default: 3
        z_size: Number of classes for classification head. Default: 1000
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
        head_init_scale: Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 768,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_type: Literal['conv', 'pool+linear'] = 'conv',
        head_init_scale: Optional[float] = None,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
            head_init_scale=head_init_scale,
        )

        # **HEAD**
        # --> Bx<z_size>
        if head_type == 'conv':
            self.head = nn.Identity()
            self.pool = nn.Sequential(
                View((-1, z_size // self.internal_num, *self.internal_wh)),
                nn.ConvTranspose2d(z_size // self.internal_num, dims[-1], kernel_size=1, stride=1),
            )
        elif head_type == 'pool+linear':
            self.head = nn.Linear(z_size, dims[-1])
            # --> Bx[3]
            self.pool = nn.Sequential(
                Norm2d(dims[-1], eps=1e-6),
                # --> Bx[3]
                Index((slice(None), slice(None), None, None)),
                # --> Bx[3]x1x1
                nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=self.internal_wh[::-1], stride=1),
                # --> Bz[3]x6x4
                Norm2d(dims[-1], eps=1e-6),  # extra, not in encoder
            )
        else:
            raise KeyError(f"head_type={head_type} not recognized")
        # --> Bx[3]x6x4

        # **UPSAMPLE LAYERS**
        # + stem and 3 intermediate upsampling conv layers
        # + 4 feature resolution stages, each consisting of multiple residual blocks
            # --> Bx[3]x6x4
        self.block3 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-1]) for _ in range(depths[3])]),
            nn.ConvTranspose2d(dims[-1], dims[-2], kernel_size=2, stride=2),
            Norm2d(dims[-2], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[2]x12x8
        self.block2 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-2]) for _ in range(depths[2])]),
            nn.ConvTranspose2d(dims[-2], dims[-3], kernel_size=2, stride=2),
            Norm2d(dims[-3], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[1]x24x16
        self.block1 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-3]) for _ in range(depths[1])]),
            nn.ConvTranspose2d(dims[-3], dims[-4], kernel_size=2, stride=2),
            Norm2d(dims[-4], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[0]x48x32
        self.block0 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-4]) for _ in range(depths[0])]),
            Norm2d(dims[-4], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[-4], in_chans, kernel_size=4, stride=4),
        )
        # --> Bx3x192x128

        # initialize weights
        self.apply(self._init_weights)
        if self.head_init_scale is not None:
            if head_type == 'pool+linear':
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        x = self.head(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.block2(x)
        x = self.block1(x)
        x = self.block0(x)
        return x


class ConvNeXtV2Ae(_Base, AeBase):

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 768,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_init_scale: Optional[float] = None,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
            head_init_scale=head_init_scale,
        )
        self.encoder = ConvNeXtV2Encoder(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
            head_init_scale=head_init_scale,
        )
        self.decoder = ConvNeXtV2Decoder(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
            head_init_scale=head_init_scale,
        )

    def _encode(self, x) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.encoder(x), []

    def _decode(self, z) -> list[torch.Tensor]:
        return [self.decoder(z)]






def convnextv2_atto(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), **kwargs)
    return model

def convnextv2ae_pico(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), **kwargs)
    return model

def convnextv2ae_nano(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), **kwargs)
    return model

def convnextv2ae_tiny(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    return model

def convnextv2ae_base(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), **kwargs)
    return model

def convnextv2ae_large(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536), **kwargs)
    return model

def convnextv2ae_huge(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(352, 704, 1408, 2816), **kwargs)
    return model



if __name__ == '__main__':

    for make_fn in [
        convnextv2_atto,
        convnextv2_femto,
        convnextv2ae_pico,
        convnextv2ae_nano,
        convnextv2ae_tiny,
        convnextv2ae_base,
        convnextv2ae_large,
        convnextv2ae_huge,
    ]:

        ae = make_fn(image_wh=(128, 192), z_size=768)

        params_enc = sum(p.numel() for p in ae.encoder.parameters())
        params_dec = sum(p.numel() for p in ae.decoder.parameters())
        params_ae = sum(p.numel() for p in ae.parameters())
        print(f"params_ae: {params_ae} ({params_ae/1_000_000:.3f}M)")
        print(f"params_enc: {params_enc} ({params_enc/1_000_000:.3f}M)")
        print(f"params_dec: {params_dec} ({params_dec/1_000_000:.3f}M)")

        x = torch.randn(1, 3, *ae.image_wh[::])
        z = torch.randn(1, ae.z_size)

        # to mps
        ae = ae.to(torch.device("mps"))
        x = x.to(torch.device("mps"))
        z = z.to(torch.device("mps"))


        def _repeat_secs(fn, sec=3):
            with tqdm() as pbar:
                start_t = last_t = time.time()
                while True:
                    t = time.time()
                    delta, last_t = t - last_t, t
                    pbar.update()
                    if t - start_t > sec:
                        break
                    fn()

        _repeat_secs(lambda: ae(x))
        _repeat_secs(lambda: ae.encoder(x))
        _repeat_secs(lambda: ae.decoder(z))
