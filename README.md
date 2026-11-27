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

| Model     | Params          | Layers |
|-----------|-----------------|--------|
| plain-20  | 269,722 (0.27M) | 20     |
| ResNet-20 | 269,722 (0.27M) | 20     |
| plain-56  | 853,018 (0.85M) | 56     |
| ResNet-56 | 853,018 (0.85M) | 56     |

## Files
- `data.py`: CIFAR-10 loaders with augmentation (pad 4, random crop, flip) and normalization
- `models.py`: `BasicBlock` (plain/residual switch, option A shortcut) and `ResNetCIFAR` (depth 6n+2)
- `train.py`: trains one model (SGD, lr schedule, AMP) and writes `log.csv`, `config.json`, `model.pt`
- `plot.py`: draws training/test error curves (`curves.png`) and the final-epoch results table (`results.md`)
- `predict.py`: loads a trained model and predicts CIFAR-10 test images or your own image
- `run_all.sh`: runs all four official trainings (plain/ResNet × 20/56) and then `plot.py`
- `check_models.py`: verifies depth, parameter counts and output shape
- `overfit_test.py`: sanity check that a model can overfit a tiny batch
- `show_images.py`: saves a grid of sample CIFAR-10 training images
- `PREDICTIONS.md`: my predictions, written before the official runs

## How to run

Full run (4 models × 164 epochs, results in `runs/`):
```bash
./run_all.sh
```

Quick check that everything works (1 epoch each, results in `runs_test/`):
```bash
EPOCHS=1 OUT=runs_test ./run_all.sh
```

Single run, e.g. plain-56:
```bash
python train.py --depth 56 --plain --epochs 164 --out runs --workers 8
python plot.py --runs runs
```

Predictions with a trained model:
```bash
# 8 random CIFAR-10 test images -> predictions.png
python predict.py --model runs/resnet56/model.pt --depth 56
# your own image -> top-3 classes
python predict.py --model runs/resnet56/model.pt --depth 56 --image cat.jpg
# plain network: add --plain
python predict.py --model runs/plain20/model.pt --depth 20 --plain
```
