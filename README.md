```markdown
# FDUNFusion-for-IVIF

This repository contains the official PyTorch implementation of the paper **"Physics-Guided Fractional-Order Deep Unfolding for Infrared and Visible Image Fusion"**.

FDUNFusion is a novel multi-source image fusion framework that integrates fractional-order calculus and deep unrolling networks (DUNs). Unlike purely data-driven methods, the weights in our network are anchored by physical priors rather than being downstream task-oriented. By explicitly modeling feature distributions, FDUNFusion effectively resolves the "staircase effect" and preserves robust visual representations across multiple datasets (MSRS, TNO, M3FD).

## 📁 Directory Structure

```text
FDUNFusion/
├── checkpoints/          # Auto-saved model checkpoints during training
├── model/                # The final best model weights (e.g., fusion_model7_..._epoch90.pth)
├── results7/             # Evaluation outputs (fused images, weight visualizations)
├── config.py             # Global configurations and default hyperparameters
├── dataset.py            # Custom dataloader for paired IR/VIS datasets
├── loss.py               # Combined fusion loss and structural constraints
├── model.py              # Core network architecture (Fractional Proximal Blocks)
├── train.py              # Training script with early-stopping and checkpointing
└── test.py               # Inference script with progress tracking and visualization

```

## ⚙️ Environment Configuration

* Python >= 3.8
* PyTorch >= 1.9.0
* torchvision
* numpy
* matplotlib
* Pillow
* tqdm

*(You can install the required packages using `pip install -r requirements.txt`)*

## 📊 Dataset Preparation

We follow a unified folder structure for datasets. The dataloader automatically matches images by natural sorting. Please organize your datasets (e.g., MSRS) as follows:

```text
data/
└── MSRS_Split/
    ├── train/
    │   ├── ir/
    │   └── vi/
    ├── val/
    │   ├── ir/
    │   └── vi/
    └── test/
        ├── ir/
        └── vi/

```

*Note: The dataset paths can be modified in `config.py` or overridden via command-line arguments.*

## 🚀 Quick Start

### 1. Testing / Inference

The `test.py` script automatically detects the optimal pre-trained weights from the `model/` directory. If absent, it will search the `checkpoints/` directory.

To test the model with default configurations on the MSRS dataset:

```bash
python test.py

```

To test on a different dataset (e.g., TNO) and evaluate specific physical prior parameters (e.g., $\lambda=0.9$, $\beta=0.3$, order = 0.1):

```bash
python test.py --order 0.1 --lambda_init 0.9 --beta_init 0.3 --ir_test_dir ./data/TNO/ir --vis_test_dir ./data/TNO/vi

```

The fused images and the attention weight visualizations will be saved in the `results7/` directory.

### 2. Training

To train the FDUNFusion network from scratch or resume from a checkpoint, run:

```bash
python train.py

```

**Hyperparameter Tuning:**
Our model supports flexible initialization for physical prior constraints. You can adjust these parameters directly via the command line:

```bash
python train.py --order 0.1 --lambda_init 0.9 --beta_init 0.3 --batch_size 8

```

* `--order`: Fractional-order degree (Default: 0.1, specifically targeted at low-frequency component extraction).
* `--lambda_init`: Initial value for the data fidelity physical prior $\lambda$.
* `--beta_init`: Initial value for the brightness constraint prior $\beta$.

**Safety Lock:** If the optimal model already exists in the `model/` folder, the script will prevent accidental overwriting. To force a completely new training session, use the `--force-retrain` flag:

```bash
python train.py --force-retrain

```

## 📝 Citation

If you find this code or our physical prior-driven framework useful for your research, please cite our paper:

```bibtex
@article{Ma2026FDUNFusion,
  title={Physics-Guided Fractional-Order Deep Unfolding for Infrared and Visible Image Fusion},
  author={Chaozhen Ma, Rencan Nie, Jinde Cao, Mingchuan Tan, Ying Zhang},
  journal={submitted（Pending determination）},
  year={2026}
}

```

```

```
