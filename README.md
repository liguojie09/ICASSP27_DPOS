# DPOS: Dual Polarity-Orbit Stem

Minimal PyTorch implementation accompanying **DPOS: A Dual-Response Stem for
Polarity-Invariant Grayscale Medical Image Classification**.

DPOS maps a grayscale image and its global intensity complement to the same stem
representation. It divides the original filter budget between signed canonical
responses and local full-wave magnitudes, followed by one backbone evaluation.

![DPOS method overview](assets/method.png)

The figure is reproduced from the manuscript. Its image, feature maps, and
probabilities use the illustrated PneumoniaMNIST example described there.

## Files

```text
DPOS_method_code/
├── dpos.py             # DPOS stem and optional convolution conversion
├── requirements.txt    # PyTorch dependency
├── README.md
└── assets/
    └── method.png      # Manuscript method figure
```

The package contains the method only: no checkpoints, dataset archives, training
scripts, or evaluation scripts.

## Installation

Use Python 3.10 or newer with PyTorch installed. Alternatively:

```bash
pip install -r requirements.txt
```

Copy `dpos.py` into your project or import it from this directory.
The implementation was checked with Python 3.13.3 and PyTorch 2.8.0 on CPU,
including bfloat16 automatic mixed precision.

## Usage

### A new grayscale stem

```python
from dpos import DPOSStem

stem = DPOSStem(out_channels=32, kernel_size=7, stride=2, padding=3)
features = stem(x)  # x: [N, 1, H, W], native 8-bit pixels / 255.0
```

The default stem has 24 signed filters and 8 magnitude filters. For input size
224 x 224, it returns features of shape `[N, 32, 112, 112]`. Keep the backbone's
normalization, activation, and subsequent layers after this stem.

### An existing input convolution

```python
from dpos import DPOSStem

# model is an existing ResNet-18; its conv1 may have grayscale or RGB inputs.
model.conv1 = DPOSStem.from_conv2d(model.conv1)
```

`from_conv2d` accepts a bias-free, ungrouped input convolution with one or three
input channels and an output width divisible by four. It preserves convolution
geometry, device, and weight dtype. RGB weights are summed into grayscale
filters. For ResNet-18, the first 48 filters initialize the signed branch and the
remaining 16 initialize the magnitude branch. Their output order is retained;
the existing normalization layer and the rest of the backbone stay in place.
The function does not download or load weights.

### Input convention

- Supply floating-point grayscale tensors of shape `[N, 1, H, W]` in `[0, 1]`,
  obtained from native 8-bit pixels divided by 255.
- Feed these values directly to DPOS; apply any backbone feature normalization
  after the stem rather than mean/std normalization before it.
- Keep the stem on the same device as its inputs. Float32 parameters can be used
  with PyTorch automatic mixed precision; integer centering and the sign
  reduction remain in float32.

## Method

For native pixels $q \in \{0,\ldots,255\}^{H\times W}$, define

$$
r=2q-255,\qquad u=r/255.
$$

The global sign uses the centered sum, with the first pixel in raster order as
the deterministic tie-breaker:

$$
\sigma(q)=
\begin{cases}
\operatorname{sign}(\sum_i r_i), & \sum_i r_i\ne0,\\
\operatorname{sign}(r_1), & \sum_i r_i=0.
\end{cases}
\qquad c(q)=\sigma(q)u(q).
$$

Every centered native 8-bit pixel is nonzero. Both sign statistics reverse under
complement, so $c(255-q)=c(q)$. The stem computes

$$
h(q)=\operatorname{Concat}\bigl(W_s*c(q),\ |W_e*u(q)|\bigr),
\qquad C_s=3C/4,\quad C_e=C/4.
$$

Both banks use bias-free convolutions. They contain the same total convolution
weights and convolution multiply-accumulate count as a one-channel stem with
the same output width and geometry. Centering, sign selection, and absolute
value add elementwise work. The native-integer identity is exact in arithmetic;
identical predictions additionally require deterministic downstream evaluation.
At the manuscript's 224 x 224 resolution, the centered integer reduction is
exactly representable in float32.

The code implements the 3:1 allocation used by DPOS24: 24/8 filters for the
compact model and 48/16 for ResNet-18. It exposes no baseline implementations.

## Results from the manuscript

The following values are transcribed from the manuscript. They describe the
paper's trained models and measurements, rather than a new benchmark of this
code package. Higher balanced accuracy (BAcc) is better.

### Compact-model comparison — Table 1

Values are mean ± sample standard deviation across three independent training
runs for PneumoniaMNIST and OrganAMNIST, and five for BreastMNIST. ERM denotes
empirical risk minimization. Each column retains the clean/inverted distinction.

| Dataset | Method | Clean BAcc ↑ | Inverted BAcc ↑ |
| --- | --- | ---: | ---: |
| PneumoniaMNIST | ERM | 0.8175 ± 0.0330 | 0.7625 ± 0.0607 |
| PneumoniaMNIST | DPOS24 | **0.8761 ± 0.0458** | **0.8761 ± 0.0458** |
| OrganAMNIST | ERM | **0.7764 ± 0.0037** | 0.3373 ± 0.0215 |
| OrganAMNIST | DPOS24 | 0.7593 ± 0.0071 | **0.7593 ± 0.0071** |
| BreastMNIST | ERM | **0.8011 ± 0.0324** | 0.5129 ± 0.0314 |
| BreastMNIST | DPOS24 | 0.7974 ± 0.0345 | **0.7974 ± 0.0345** |

Bold indicates the higher mean between the two methods shown for each dataset
and column. The manuscript also reports two-view test-time augmentation (TTA).

### ResNet-18 forward latency — Table 5

Batch size 32, float32, NVIDIA A800; lower is better. After 10 warm-ups, the paper
reports the median of five groups of 20 synchronized timed evaluations.

| Method | Backbone passes | PneumoniaMNIST (ms) ↓ | OrganAMNIST (ms) ↓ | BreastMNIST (ms) ↓ |
| --- | ---: | ---: | ---: | ---: |
| ERM | 1 | **6.985** | **7.087** | **7.001** |
| ERM + TTA | 2 | 14.000 | 14.186 | 14.047 |
| DPOS24 | 1 | 7.540 | 7.662 | 7.598 |

DPOS24 uses approximately 54% of the measured two-view forward latency. Table 6
reports zero measured clean/complement prediction discrepancy for DPOS24 on all
three ResNet-18 test sets.

## Code availability in the paper

After uploading this directory to your repository, its public URL can be added
to the abstract with the sentence: “The method implementation is available at
[repository URL].” Replace the placeholder with the actual repository URL.
