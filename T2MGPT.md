# T2M-GPT — Text-to-IMU Experimental Pipeline

> **Status: Experimental.** This pipeline is not part of the main framework.

## Overview

This pipeline generates synthetic IMU data from a natural-language description of a movement. A text prompt is fed into T2M-GPT (a VQ-VAE-based motion generator), which produces a 3D motion sequence. That sequence then passes through the same IMU synthesis chain used by Pipeline 1.

```
Text prompt → T2M-GPT (VQ-VAE decode) → Pose3D → MetricNormalizer
           → RootEstimator → IMUSim → virtual_imu.npz
```

## Additional Dependencies

This pipeline requires pre-trained weights that are **not downloaded automatically** and must be placed manually:

- `pretrained/VQVAEV3_CB1024_CMT_H1024_NRES3/` — VQ-VAE decoder weights
- `checkpoints/t2m/` — mean/std normalization statistics

The relevant code lives in `pose_module/prompt_source/` and `t2mgpt/`.

## Usage

```bash
source .venv/bin/activate

python -m pose_module.prompt_source.generate \
  --prompt "A person walks forward and waves" \
  --output-dir output/text2imu
```

Output lands under `output/text2imu/` following the same per-clip structure as the other pipelines:
- `pose/pose3d/pose3d.npz` — generated 3D skeleton
- `imu/virtual_imu.npz` — synthetic accelerometer + gyroscope

## Key Modules

- `pose_module/prompt_source/` — T2M-GPT backend and generate CLI entry point
- `t2mgpt/` — VQ-VAE weights and inference code
