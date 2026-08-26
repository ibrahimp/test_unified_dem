import torch
import torch.nn as nn
from pathlib import Path

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedNAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, gamma, beta):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.conv2(y)
        x = x + self.dropout1(y) * gamma

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        return x + self.dropout2(y) * beta

class SimplifiedNAFNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, width=32, blocks=16, downsample=4, dropout=0.0, debug_logging=False, binning=1):
        super().__init__()
        
        # --- BOTTLENECK RESTORED: Spatial downsampling at input ---
        self.downsample = downsample
        self.in_proj = nn.Conv2d(in_channels, width, downsample, stride=downsample)
        
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, blocks * 2 * width)
        )
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
            biases = torch.zeros(blocks * 2 * width)
            for i in range(blocks):
                gamma_start = (2 * i) * width
                beta_start = (2 * i + 1) * width
                biases[gamma_start : gamma_start + width] = 1.0
                biases[beta_start : beta_start + width] = 1.0
            self.mlp[-1].bias.copy_(biases)

        self.body = nn.ModuleList([SimplifiedNAFBlock(width, dropout=dropout) for _ in range(blocks)])
        
        # --- BOTTLENECK RESTORED: SR upsampling at output ---
        self.out_proj = nn.Conv2d(width, out_channels * downsample * downsample, 1)
        self.upsample = nn.PixelShuffle(downsample)
        
        # Debug logging control
        self.debug_logging = debug_logging
        if self.debug_logging:
            self.log_path = Path("runs/debug_logs")
            self.log_path.mkdir(parents=True, exist_ok=True)
            self.batch_count = 0
            self.LOG_INTERVAL = 100
        else:
            self.batch_count = 0

    def forward(self, x, noise_level):
        # FIX 4: Robust noise_level shape handling
        if not isinstance(noise_level, torch.Tensor):
            z = torch.tensor([[noise_level]], device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        else:
            if noise_level.dim() == 0:
                z = noise_level.view(1, 1, 1, 1).expand(x.shape[0], -1, -1, -1)
            elif noise_level.dim() == 1:
                if noise_level.shape[0] != x.shape[0]:
                    z = noise_level.expand(x.shape[0]).view(-1, 1, 1, 1)
                else:
                    z = noise_level.view(-1, 1, 1, 1)
            else:
                try:
                    z = noise_level.reshape(-1, 1, 1, 1)
                except RuntimeError:
                    z = noise_level.squeeze().view(-1, 1, 1, 1)
        
        y = self.in_proj(x)
        
        z_flat = z.view(-1, 1)
        params = self.mlp(z_flat)
        
        params = params.view(-1, len(self.body) * 2, self.in_proj.out_channels)
        
        if self.debug_logging and self.batch_count % self.LOG_INTERVAL == 0:
            with torch.no_grad():
                all_gammas = params[:, 0::2, :].mean(dim=0)
                all_betas = params[:, 1::2, :].mean(dim=0)
                
                with open(self.log_path / "film_evolution.txt", "a") as f:
                    f.write(f"Batch {self.batch_count} | Gamma_means: {all_gammas.detach().cpu().numpy().flatten()}\n")
                    f.write(f"Batch {self.batch_count} | Beta_means:  {all_betas.detach().cpu().numpy().flatten()}\n")
        
        self.batch_count += 1

        for i, block in enumerate(self.body):
            gamma = params[:, i*2, :].view(-1, self.in_proj.out_channels, 1, 1)
            beta = params[:, i*2+1, :].view(-1, self.in_proj.out_channels, 1, 1)
            y = block(y, gamma, beta)
            
        y = self.out_proj(y)
        y = self.upsample(y) # Restore latent-space upsampling
        
        # --- PURE LINEAR OUTPUT (No Sigmoid/Clamp) ---
        return y

def make_input_channels(args):
    channels = 1
    if args.use_cfa:
        channels += 3
    return channels

def build_model(args, debug_logging=False):
    return SimplifiedNAFNet(
        make_input_channels(args),
        width=args.width,
        blocks=args.blocks,
        downsample=args.downsample,
        dropout=0.0,
        debug_logging=debug_logging,
        binning=args.binning,
    )
