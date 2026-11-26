# ResNet vs Plain on CIFAR-10

Reimplementation of He et al. (2016), "Deep Residual Learning for Image Recognition".
Comparing plain and residual networks at depth 20 and 56 under the same training budget.

## Setup (paper §4.2)

| Setting       | Paper                                   | This repo                         |
|---------------|-----------------------------------------|-----------------------------------|
| Depth         | 6n+2 (n=3 → 20, n=9 → 56)               | same                              |
| Channels      | 16 / 32 / 64, maps 32 / 16 / 8          | same                              |
| Shortcut      | option A (identity + zero-padding)      | same                              |
| Init          | He (Kaiming)                            | same                              |
| Optimizer     | SGD, lr 0.1, momentum 0.9, wd 1e-4      | same                              |
| Batch size    | 128                                     | same                              |
| Schedule      | lr ÷10 at 32k, 48k iters, stop at 64k   | ÷10 at epoch 82, 123, stop at 164 |
| Augmentation  | pad 4, random crop 32, horizontal flip  | same                              |
| Normalization | per-pixel mean subtraction              | per-channel mean/std (deviation)  |
| Precision     | fp32                                    | mixed precision, AMP (deviation)  |

## Model check

| Model     | Params  | Layers |
|-----------|---------|--------|
| plain-20  | 0.272M  | 20     |
| ResNet-20 | 0.272M  | 20     |
| plain-56  | 0.856M  | 56     |
| ResNet-56 | 0.856M  | 56     |

## Files
- `data.py`: CIFAR-10 loaders with augmentation
- `models.py`: `BasicBlock` (plain/residual switch, option A shortcut) and `ResNetCIFAR`
- `check_models.py`: verifies depth, parameter counts and output shape