# U-Net Brain Tumour MRI Segmentation

Brain-tumour segmentation on MRI scans with a **U-Net implemented from scratch in pure NumPy** — no PyTorch, no TensorFlow, no GPU. Encoder–decoder with skip connections, trained with Adam on synthetic brain MRIs, evaluated with Dice and IoU.

**Verified result (deterministic, reproduced in a fresh venv): validation Dice 0.744, held-out Dice 0.714** after 8 epochs on 48 training / 16 validation images. 11/11 tests pass.

## Why this project is different

- **A U-Net with no deep-learning framework.** Every layer — convolution (im2col), max-pooling, upsampling, ReLU, sigmoid — is hand-written in `src/brainseg/nn.py`, including the backward pass and Adam. It runs anywhere Python runs.
- **The class-imbalance fix is the lesson.** Tumour pixels are ~4% of a scan, so an unweighted loss learns to predict background everywhere (loss falls, Dice = 0.000). A positive-class-weighted BCE (×20) fixes it: Dice jumps to 0.744.
- **No patient data needed.** Real scans can't ship in a public repo, so `src/brainseg/data.py` synthesises realistic MRIs: elliptical brain, darker ventricles, irregular lumpy tumours with an oedema ring, a smooth bias field, and noise. Everything is seeded and reproducible.

## Features

- Synthetic brain-MRI generator (brain, ventricles, tumour + oedema, bias field, noise)
- Pure-NumPy U-Net: 29,617 parameters, encoder–decoder with skip connections
- Training with Adam and class-weighted BCE loss
- Dice / IoU evaluation and visual overlay reports (MRI + ground truth + prediction)
- Deterministic end-to-end demo with a real output file
- 11/11 tests green, verified twice — including once in a brand-new venv

## Architecture

```
Input 1×64×64
  enc1: conv 1→8, conv 8→8, max-pool        (skip 8 ch)
  enc2: conv 8→16, conv 16→16, max-pool     (skip 16 ch)
  bottleneck: conv 16→32, conv 32→32
  dec2: upsample, concat skip2, conv 48→16, conv 16→16
  dec1: upsample, concat skip1, conv 24→8,  conv 8→8
  head: 1×1 conv 8→1, sigmoid → tumour probability map
```

## Installation

```bash
git clone https://github.com/murtazamahboob/unet-brain-tumor-segmentation.git
cd unet-brain-tumor-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train and save the model + history
brainseg train --epochs 8

# Full demo: train, evaluate on held-out data, write the visual report
python examples/demo.py
# -> outputs/demo_report.md, outputs/overlay.png, outputs/model.npz
```

## Example output

```
epoch  1: train_loss=1.0778 val_loss=0.4116 val_dice=0.681
epoch  8: train_loss=0.1745 val_loss=0.0531 val_dice=0.744
demo done: held-out Dice=0.714  report=outputs/demo_report.md
```

Running the demo writes `outputs/overlay.png` — a 4-row figure with the MRI slice, ground-truth tumour (green) and predicted tumour (red) for held-out cases, each labelled with its Dice score (0.66–0.78 in the verified run).

## Project structure

```
src/brainseg/
  data.py       # synthetic brain-MRI + tumour-mask generation
  nn.py         # pure-NumPy layers: Conv2d, MaxPool, Upsample, ReLU, Sigmoid
  unet.py       # U-Net architecture, Adam optimiser, weighted BCE loss
  metrics.py    # Dice coefficient, IoU
  train.py      # training loop
  visualize.py  # MRI / ground-truth / prediction overlay figures
  cli.py        # `brainseg train` / `brainseg demo`
tests/          # 11 tests: data, metrics, layers, U-Net, training step
examples/demo.py
outputs/        # demo_report.md, overlay.png, model.npz (generated, not committed)
```

## Tech stack

Python 3.10+ · NumPy · SciPy · Matplotlib · Pillow · pytest

## License

MIT — see [LICENSE](LICENSE).
