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

class ResNetCIFAR(nn.Module):
    """The full network from paper §4.2. depth = 6n + 2. residual=False → plain network."""

    def __init__(self, depth=20, residual=True, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (20, 32, 44, 56, ...)"
        n = (depth - 2) // 6                       # blocks per stage: 3 for 20, 9 for 56

        # first layer: 3 colors -> 16 channels                         (1 layer)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # three stages of n blocks each                                (6n layers)
        self.stage1 = self._make_stage(16, 16, n, stride=1, residual=residual)  # 32x32
        self.stage2 = self._make_stage(16, 32, n, stride=2, residual=residual)  # 16x16
        self.stage3 = self._make_stage(32, 64, n, stride=2, residual=residual)  # 8x8

        # final layer: 64 numbers -> 10 class scores                   (1 layer)
        self.fc = nn.Linear(64, num_classes)

        # He (Kaiming) initialization for conv layers, as in the paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_stage(self, in_ch, out_ch, n, stride, residual):
        # first block may shrink the image / add channels, the rest keep the shape
        blocks = [BasicBlock(in_ch, out_ch, stride, residual)]
        for _ in range(n - 1):
            blocks.append(BasicBlock(out_ch, out_ch, 1, residual))
        return nn.Sequential(*blocks)               # chain the blocks in order

    def forward(self, x):                           # x: [B, 3, 32, 32]
        out = F.relu(self.bn1(self.conv1(x)))       # [B, 16, 32, 32]
        out = self.stage1(out)                      # [B, 16, 32, 32]
        out = self.stage2(out)                      # [B, 32, 16, 16]
        out = self.stage3(out)                      # [B, 64, 8, 8]
        out = F.adaptive_avg_pool2d(out, 1)         # [B, 64, 1, 1]  average each 8x8 map
        out = out.flatten(1)                        # [B, 64]
        return self.fc(out)                         # [B, 10]  class scores