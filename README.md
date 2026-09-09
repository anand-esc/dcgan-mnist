# DCGAN on Fashion-MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anand-esc/dcgan-mnist/blob/main/dcgan_colab.ipynb)

**Deep Convolutional Generative Adversarial Network** — Clean PyTorch implementation with industry-standard stabilization techniques. Generates realistic fashion items from noise in ~5 minutes on free Colab GPU.

---

## Quick Start

```bash
# Local
python main.py train

# Colab (5 min on T4)
# 1. Upload project → Runtime → GPU (T4) → Run all cells in dcgan_colab.ipynb
```

**Output:** `checkpoints/checkpoint_final.pth` + `samples/` with generated grids & loss curves.

---

## What This Does

| Input | Output |
|-------|--------|
| Random noise `z ~ N(0,1)` [100-dim] | 64×64 grayscale fashion images |
| Fashion-MNIST (60K training images) | Generator learns data distribution |

**Architecture:** DCGAN (Radford et al., 2015) — fully convolutional, no pooling, BatchNorm throughout, ReLU/LeakyReLU activations, Tanh output normalized to [-1, 1].

---

## Stabilization Techniques (Built-In)

| Technique | Config | Purpose |
|-----------|--------|---------|
| **Label Smoothing** | `label_smoothing: 0.1` | Real=0.9, Fake=0.1 — prevents D overconfidence |
| **Batch Normalization** | All layers (except G-in/D-out) | Stabilizes gradients, reduces covariate shift |
| **Tuned Adam** | `lr=2e-4, β₁=0.5, β₂=0.999` | Lower β₁ reduces momentum oscillation |
| **Weight Init** | Normal(0, 0.02) / BN(1, 0.02) | Critical for GAN convergence |
| **Fixed Noise Tracking** | 64 samples/epoch | Visual progress comparison |

---

## Project Structure

```
dcgan-mnist/
├── main.py                    # Entry: train / evaluate
├── configs/config.json        # All hyperparameters
├── requirements.txt           # 7 dependencies
├── src/
│   ├── models.py              # Generator & Discriminator
│   ├── train.py               # Training loop + logging
│   ├── evaluate.py            # Grids, interpolation, mode collapse analysis
│   └── data_utils.py          # Fashion-MNIST / MNIST loaders
├── checkpoints/               # Auto-created: .pth files
├── samples/                   # Auto-created: grids, loss_curves.png
├── logs/                      # Auto-created: TensorBoard events
├── data/                      # Auto-created: downloaded datasets
├── dcgan_colab.ipynb          # Colab-ready notebook
└── COLAB_README.md            # Colab quick guide
```

---

## Configuration (`configs/config.json`)

```json
{
  "dataset": "fashion-mnist",   // or "mnist"
  "data_root": "./data",
  "batch_size": 128,
  "image_size": 64,
  "nc": 1,                      // channels (1 for MNIST)
  "nz": 100,                    // latent dim
  "ngf": 64,                    // generator base features
  "ndf": 64,                    // discriminator base features
  "num_epochs": 50,
  "lr": 0.0002,
  "beta1": 0.5,
  "beta2": 0.999,
  "label_smoothing": 0.1,
  "ngpu": 1,
  "checkpoint_interval": 5,
  "sample_interval": 100,
  "log_interval": 50,
  "num_workers": 4,
  "seed": 42
}
```

**Key knobs:** `batch_size` (VRAM), `lr` (stability), `beta1` (critical=0.5), `label_smoothing` (0.05-0.2).

---

## Commands

```bash
# Train from scratch
python main.py train

# Resume from checkpoint
python main.py train --resume checkpoints/checkpoint_epoch_0025.pth

# Generate sample grid (64 images)
python main.py evaluate --checkpoint checkpoints/checkpoint_final.pth

# Latent space interpolation (5 pairs × 10 steps)
python main.py evaluate --checkpoint checkpoints/checkpoint_final.pth --interpolation

# Sample sheet: 8 samples from every checkpoint epoch
python main.py evaluate --sample_sheet

# Mode collapse analysis (pairwise distances)
python -c "from src.evaluate import analyze_mode_collapse; analyze_mode_collapse('samples/individual', 500)"
```

---

## Monitoring

```bash
# TensorBoard (local)
tensorboard --logdir logs --port 6006

# In Colab notebook: %tensorboard --logdir logs
```

**Tracked metrics:** Generator/Discriminator loss (step + epoch), D(x), D(G(z)) before/after G update.

**Healthy signs:** D(x) ≈ 0.5, D(G(z)) ≈ 0.5, losses oscillate downward.

---

## Expected Results (Fashion-MNIST, 50 epochs, T4)

| Epoch | Visual Quality | Typical Losses |
|-------|----------------|----------------|
| 5 | Blurry shapes | G~2.3, D~1.2 |
| 15 | Recognizable items | G~1.8, D~1.0 |
| 30 | Clear textures | G~1.5, D~0.7 |
| 50 | Sharp, diverse | G~1.3, D~0.5 |

**Mode collapse check:** Mean pairwise distance > 0.15 = healthy.

---

## Colab (Free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU (T4)**
3. Upload `dcgan_ibm_project_colab` folder (zip → unzip)
4. Open `dcgan_colab.ipynb` → **Runtime → Run all**
5. **Download `dcgan_results.zip` before session ends**

Timeline: ~30s setup → 5-8 min training → 10s samples → 5s download.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| D_loss→0, G_loss→∞ | D too strong | Increase label_smoothing, reduce D LR |
| Mode collapse | Low diversity | Increase label_smoothing, check BN in train() |
| NaN loss | Numerical instability | Reduce LR to 1e-4, add gradient clipping |
| CUDA OOM | VRAM full | Reduce batch_size (128→64→32) |
| Slow training | CPU bottleneck | Increase num_workers, pin_memory=True |

---

## Extending

- **WGAN-GP** — Replace BCE with Wasserstein + gradient penalty
- **Conditional GAN** — Add class labels to G and D
- **Higher resolution** — Add ConvTranspose layers for 128×128
- **CelebA faces** — Set `nc: 3`, use CelebA dataset (~1.4GB)
- **Mixed precision** — `torch.cuda.amp` for 2× speedup

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
pillow>=9.5.0
```

Python 3.8+, CUDA 11.7+ for GPU (optional).

---

## License

MIT — See [LICENSE](LICENSE).

---

## Reference

Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434