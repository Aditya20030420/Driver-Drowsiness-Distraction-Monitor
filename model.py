"""DriverMonitorNet: CNN (per-frame spatial features) -> GRU (30-frame temporal state) -> 3-way classifier.

Tensor shape journey (B=batch, T=30 frames, C=1, H=W=64):

    input            (B, T, 1, 64, 64)
    contiguous view  (B*T, 1, 64, 64)      <- time folded into batch so Conv2d sees plain images
    conv block x4    (B*T, 128, 4, 4)      <- 64 -> 32 -> 16 -> 8 -> 4 via MaxPool2d(2)
    flatten + Linear (B*T, 512)            <- dense spatial vector per frame
    view             (B, T, 512)           <- time dimension reconstructed for the GRU
    GRU              h_n: (layers, B, 128) <- last layer's final hidden state = (B, 128)
    Linear + Softmax (B, 3)                <- [Alert, Drowsy, Distracted]
"""
import torch
import torch.nn as nn

CLASSES = ("Alert", "Drowsy", "Distracted")
SEQ_LEN = 30
PATCH = 64


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DriverMonitorNet(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 512, hidden: int = 128, n_classes: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(_block(in_ch, 16), _block(16, 32), _block(32, 64), _block(64, 128))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, feat_dim), nn.ReLU(inplace=True))
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)  # batch_first -> (B, T, feat)
        self.head = nn.Linear(hidden, n_classes)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """x: (B, T, C, H, W) float in [0, 1]. Returns softmax probs (B, 3), or raw logits for CrossEntropyLoss."""
        b, t, c, h, w = x.shape
        x = x.contiguous().view(b * t, c, h, w)          # (B*T, C, H, W)
        f = self.proj(self.cnn(x))                       # (B*T, 512)
        f = f.view(b, t, self.feat_dim)                  # (B, T, 512)
        _, h_n = self.gru(f)                             # h_n: (1, B, hidden)
        logits = self.head(h_n[-1])                      # (B, 3)
        return logits if return_logits else torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    net = DriverMonitorNet().eval()
    out = net(torch.rand(2, SEQ_LEN, 1, PATCH, PATCH))
    assert out.shape == (2, 3) and torch.allclose(out.sum(-1), torch.ones(2)), out
    print("ok", out.shape, f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
