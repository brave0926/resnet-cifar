# Predictions (written before the official runs, 2026-11-27)

> Note: written on 2026-11-27, before the official plain vs. ResNet runs. The ResNet-20 pilot result (8.18% test error, laptop RTX 3050) was already known; no plain network had been trained yet.

## 1. Depth 20: plain-20 vs ResNet-20
- Which will have lower test error? By how much (more or less than 1%p)?
  - ResNet-20 will be slightly better, 
    but the gap will be small, around 1%p or less, so it may count as "no difference".
- Why?
  - 20 layers is shallow enough that a plain network can still be optimized well. 
    The degradation problem appears mainly in deeper networks. The paper shows the same on ImageNet: 
    at 18 layers, plain and ResNet are almost equal (27.94% vs 27.88%).

## 2. Depth 56: plain-56 vs ResNet-56
- Which will have lower test error? By how much?
  - ResNet-56 will be clearly better, by several %p (well above 1%p).
- Why?
  - In a 56-layer plain network, every layer must learn a useful transformation and the training
    signal has to pass back through all 56 layers, so optimization becomes hard. 
    In ResNet, each block only learns a correction F(x) on top of x. An unneeded block can stay close to identity,
    and the shortcuts give the gradient a direct path to the early layers.

## 3. Does depth help?
- plain-20 vs plain-56: which is better?
  - plain-20 will be better. Adding layers to a plain network makes it worse (degradation).
- ResNet-20 vs ResNet-56: which is better?
  - ResNet-56 will be better. With shortcuts, the extra layers can add useful corrections without hurting, 
    so depth helps again (paper: 8.75% → 6.97%).

## 4. Training error (the degradation test)
- Will plain-56 have HIGHER training error than plain-20?
  - Yes, plain-56 will have a higher training error than plain-20.
- Why does this matter (overfitting vs optimization problem)?
  - If the deeper network were only overfitting, its training error would be lower and only the test error higher.
    A higher training error means the deeper plain network cannot even fit the training data, so the problem is optimization, 
    not overfitting. This is the main claim of the paper (Fig. 1 and Fig. 6).

## 5. Rough numbers I expect
| Model     | Final test error (%) |
|-----------|----------------------|
| plain-20  | ~9.0                 |
| ResNet-20 | ~8.2                 |
| plain-56  | ~11–13               |
| ResNet-56 | ~7.0                 |

Basis: pilot ResNet-20 = 8.18%; paper Table 6 (ResNet-20 8.75%, ResNet-56 6.97%); paper Fig. 6 (plain-56 noticeably worse than plain-20).