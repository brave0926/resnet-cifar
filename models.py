import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """One block = 2 conv layers. With residual=True it adds the shortcut."""

    def __init__(self, in_ch, out_ch, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        # --- the boxes (layers) ---
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def shortcut(self, x):                                              # ← NEW
        # Option A: if the shape changes, make x match WITHOUT adding weights
        if self.stride != 1 or self.in_ch != self.out_ch:
            x = x[:, :, ::2, ::2]                     # keep every 2nd pixel: 32x32 -> 16x16
            extra = self.out_ch - self.in_ch          # missing channels: 32 - 16 = 16
            x = F.pad(x, (0, 0, 0, 0, extra // 2, extra - extra // 2))  # add zero channels
        return x

    def forward(self, x):
        # --- the arrows (order of computation) ---
        out = self.conv1(x)              # conv 3x3
        out = F.relu(self.bn1(out))      # BN + ReLU
        out = self.conv2(out)            # conv 3x3
        out = self.bn2(out)              # BN
        if self.residual:
            out = out + self.shortcut(x)  # shortcut (reshaped if needed)  ← CHANGED
        return F.relu(out)               # final ReLU


if __name__ == "__main__":
    x = torch.randn(2, 16, 32, 32)       # 2 fake images, 16 channels, 32x32

    plain = BasicBlock(16, 16, stride=1, residual=False)
    res = BasicBlock(16, 16, stride=1, residual=True)
    print("plain block output:", plain(x).shape)
    print("res block output:  ", res(x).shape)

    count = lambda m: sum(p.numel() for p in m.parameters())
    print("plain params:", count(plain), "| res params:", count(res))

    down = BasicBlock(16, 32, stride=2, residual=True)
    print("down block output:", down(x).shape)