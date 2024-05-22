import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from xformers.ops import memory_efficient_attention


class Attention(nn.Module):
    def __init__(
        self,
        heads: int,
        emb_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // self.heads
        assert self.emb_dim % self.heads == 0
        self.qkv = nn.Conv2d(emb_dim, 3 * emb_dim, 1, bias=False)
        self.out = nn.Conv2d(emb_dim, emb_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        q, k, v = rearrange(
            self.qkv(x), "b (qkv h d) x y -> qkv b (x y) h d", h=self.heads, qkv=3
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = memory_efficient_attention(q, k, v)
        out = rearrange(out, "b (x y) h d -> b (h d) x y", x=h, y=w)
        out = self.out(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim: int, dim_mult: int = 4) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, dim * dim_mult)
        self.w2 = nn.Linear(dim * dim_mult, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        x = rearrange(x, "b h x y -> b (x y) h")
        x = self.w2(F.silu(self.w1(x)))
        return rearrange(x, "b (x y) h -> b h x y", x=h, y=w)


class AttentionBlock(nn.Module):
    def __init__(self, heads: int, channels: int) -> None:
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, channels)
        self.groupnorm2 = nn.GroupNorm(32, channels)
        self.attention = Attention(heads, channels)
        self.ffn = FFN(channels, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.groupnorm1(x))
        x = x + self.ffn(self.groupnorm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.activation = nn.SiLU(True)
        if in_channels == out_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        x = self.groupnorm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.groupnorm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x + self.skip_layer(skip)


class Encoder(nn.Module):
    def __init__(self, h_dim: int, in_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, 512),
            AttentionBlock(8, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(8, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, h_dim, kernel_size=3, padding=1),
            nn.Conv2d(h_dim, h_dim, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self, h_dim: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=1, padding=0),
            nn.Conv2d(h_dim, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(8, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(8, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, out_ch, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class VQVAEQuantize(nn.Module):
    def __init__(self, num_hiddens: int, n_embed: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.temperature = 1.0
        self.kld_scale = 0.0

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z_e: torch.Tensor, hard = False: bool) -> torch.Tensor:
        z_e = self.proj(z_e)
        one_hot = F.gumbel_softmax(z_e, tau=self.temperature, dim=1, hard=hard)
        z_q = torch.einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        qy = F.softmax(z_e, dim=1)
        loss = (
            self.kld_scale
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )
        return z_q, loss


class VQVae(nn.Module):
    def __init__(
        self,
        in_ch: int,
        h_dim: int,
        n_embed: int,
        emb_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(h_dim, in_ch)
        self.decoder = Decoder(h_dim, in_ch)
        self.vector_quantizer = VQVAEQuantize(h_dim, n_embed, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        z_q, latent_loss = self.vector_quantizer(z_e)
        x_h = self.decoder(z_q)
        recon_loss = F.mse_loss(x_h, x)
        loss = recon_loss + latent_loss
        return loss

    def params_count(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
