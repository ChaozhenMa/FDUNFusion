# FDUNFusion-for-IVIF

This repository provides the official PyTorch implementation of **“Fractional-Order Deep Unfolding with Prior-Regularized Spatial Weighting for Infrared and Visible Image Fusion.”**

FDUNFusion combines explicit model-based updates with learned residual corrections to integrate thermal saliency and structural details from infrared and visible images.

## Overview

The network consists of a multi-stage unfolding architecture and a stage-shared spatial Weight-Net:

- **Explicit data-fidelity updates:** Each stage updates the fused luminance using complementary spatial weights for the two source modalities.
- **Fractional residual refinement:** A stage-specific Fractional Residual Block (FRB) uses signed directional discrepancies derived from finite one-sided backward Grünwald–Letnikov (GL) differences. It predicts a bounded residual correction modulated by a source-conditioned gate.
- **Prior-regularized spatial weighting:** Weight-Net predicts spatial weights from the source luminances. During training, these weights are regularized by reference maps constructed from global luminance contrast and normalized intensity. The reference maps are not required at inference.
- **Luminance-domain fusion:** The network predicts fused luminance and retains visible-image chrominance for color reconstruction.

The architecture is optimization-inspired; the FRB is a learned residual correction and is not assumed to be an exact proximal operator.

Experiments on **MSRS, TNO, and M³FD** demonstrate competitive fusion quality. Additional evaluations with fixed semantic segmentation and object detection models assess downstream compatibility without method-specific retraining.

## Directory Structure

```text
FDUNFusion/
├── checkpoints/          # Training checkpoints
├── model/                # Pretrained model weights
├── results7/             # Fused images and weight visualizations
├── config.py             # Paths, hyperparameters, and experiment settings
├── dataset.py            # Paired infrared/visible image loading
├── loss.py               # Joint fusion and weight-regularization objective
├── model.py              # Unfolding stages, FRBs, GL operators, and Weight-Net
├── train.py              # Training entry point
└── test.py               # Inference entry point
```

## Environment

Required packages:

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision
- numpy
- matplotlib
- Pillow
- tqdm

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize paired infrared and visible images as follows:

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

Each infrared image must have a corresponding, spatially registered visible image with matching spatial dimensions. Ensure that filenames identify the correct image pairs.

For training, paired images must use the same spatial crop coordinates to preserve registration.

Configure the dataset paths in `config.py`. Inference input directories can also be specified through command-line arguments.

Use the validation split for hyperparameter selection and keep the test split separate from model selection.

## Inference

Place the pretrained weights in the location expected by the configuration and inference script. Ensure that the selected checkpoint corresponds to the intended experiment.

Run inference with the default configuration:

```bash
python test.py
```

To specify an experiment and input directories:

```bash
python test.py \
    --order 0.1 \
    --lambda_init 0.9 \
    --beta_init 0.3 \
    --ir_test_dir ./data/TNO/ir \
    --vis_test_dir ./data/TNO/vi
```

For Windows PowerShell, use the command on one line:

```powershell
python test.py --order 0.1 --lambda_init 0.9 --beta_init 0.3 --ir_test_dir ./data/TNO/ir --vis_test_dir ./data/TNO/vi
```

Check the console output to confirm that the intended pretrained checkpoint was loaded successfully before using the fusion results.

Outputs are saved to the experiment-specific directory configured in `config.py`, under `results7/` in the provided project structure.

Changing the experiment arguments does not replace training or checkpoint selection. Evaluate each parameter configuration using its corresponding trained checkpoint.

## Training

Configure the training and validation paths in `config.py`, then run:

```bash
python train.py
```

Example with explicit parameter settings:

```bash
python train.py --order 0.1 --lambda_init 0.9 --beta_init 0.3 --batch_size 8
```

### Parameter Definitions

- `--order`: Fractional order α used by the GL directional difference operators.
- `--lambda_init`: Initial value of the stage-wise fractional residual gain λₖ. In the coupled configuration used in this implementation, it also sets the fractional loss coefficient.
- `--beta_init`: Initial value of the stage-wise brightness parameter βₖ. In the coupled configuration used in this implementation, it also sets the brightness loss coefficient.
- `--batch_size`: Number of paired samples in each training mini-batch.

The stage-wise parameters are learned during training. The associated loss coefficients are fixed by the experiment configuration.

These settings control the fractional representation, stage initialization, and loss weighting. They are not physical priors.

## Evaluation

The manuscript evaluates fusion quality on MSRS, TNO, and M³FD.

For downstream evaluation, fused images are processed by the same fixed SegFormer and YOLOv9-C evaluators across fusion methods. These experiments assess compatibility with fixed downstream models; the downstream networks are not used to train FDUNFusion.

The MSRS detection evaluation subset contains 80 images and 193 annotated instances: 144 persons and 49 cars. No bicycle ground-truth instances are present in this subset, so bicycle AP is reported as **N/A**.

## Citation

If you use this implementation in your research, please cite the manuscript:

```bibtex
@unpublished{Ma2026FDUNFusion,
  title  = {Fractional-Order Deep Unfolding with Prior-Regularized Spatial Weighting for Infrared and Visible Image Fusion},
  author = {Ma, Chaozhen and Nie, Rencan and Cao, Jinde and Tan, Mingchuan and Zhang, Ying},
  note   = {Unpublished manuscript},
  year   = {2026}
}
```

The citation will be updated when publication details become available.
